package relay

import (
	"encoding/json"
	"strings"

	"github.com/QuantumNous/new-api/relaykit/dto"
	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
)

// 长上下文兜底压缩配置：仅对 deepseek-v4-flash 且路由到 ch21-23（BAI）的请求生效。
// 上游上下文上限约 360K，客户端可能发送远超该值的上下文；网关在转发前做
// 「保留最近完整轮次 + 早期历史合并」的兜底压缩，避免超限请求被上游拒绝。
const (
	compressTriggerTokens = 280000 // 请求上下文超过该值触发压缩（留足余量）
	compressTargetTokens  = 250000 // 压缩目标：压缩后不超过该值
	compressKeepRounds    = 15     // 保留最近完整对话轮数
	compressModelName     = "deepseek-v4-flash"
	compressChannelMinID  = 21
	compressChannelMaxID  = 23
)

// estimateTextTokens 保守估算文本 token 数。
// 采用 UTF-8 字节数/3：中文约 1 token/字、英文约 1.33 token/字，整体偏保守，
// 宁可多压缩也不让超限请求撞上游 360K 硬上限。
func estimateTextTokens(s string) int {
	if len(s) == 0 {
		return 0
	}
	return len(s)/3 + 1
}

// ShouldCompressContext 判断请求是否命中长上下文压缩范围（flash + ch21-23）。
func ShouldCompressContext(channelID int, modelName string) bool {
	if channelID < compressChannelMinID || channelID > compressChannelMaxID {
		return false
	}
	return strings.TrimSpace(modelName) == compressModelName
}

// CompressChatMessages 对 chat 协议消息做兜底压缩：保留最近完整轮次 + 早期历史合并。
// 总 token 未超阈值时原样返回。
func CompressChatMessages(messages []dto.Message) []dto.Message {
	if len(messages) == 0 {
		return messages
	}
	total := 0
	for i := range messages {
		total += estimateTextTokens(messages[i].Role) + estimateTextTokens(messages[i].StringContent())
	}
	if total <= compressTriggerTokens {
		return messages
	}

	// 分离 system 消息（保留）与普通历史。
	var system []dto.Message
	history := make([]dto.Message, 0, len(messages))
	for i := range messages {
		if messages[i].Role == "system" {
			system = append(system, messages[i])
		} else {
			history = append(history, messages[i])
		}
	}

	// 从后往前定位第 keepRounds 个 user 消息，其后的全部内容视为最近完整轮次。
	keepStart := 0
	userCount := 0
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "user" {
			userCount++
			if userCount == compressKeepRounds {
				keepStart = i
				break
			}
		}
	}
	recent := history[keepStart:]
	early := history[:keepStart]

	// 早期历史合并为一条 user 消息。
	var merged strings.Builder
	for i := range early {
		if text := strings.TrimSpace(early[i].StringContent()); text != "" {
			merged.WriteString(early[i].Role)
			merged.WriteString(": ")
			merged.WriteString(text)
			merged.WriteString("\n")
		}
	}

	out := make([]dto.Message, 0, len(system)+len(recent)+1)
	out = append(out, system...)
	if merged.Len() > 0 {
		var mergedMsg dto.Message
		mergedMsg.Role = "user"
		mergedMsg.SetStringContent("[以下为此前对话的合并历史]\n" + merged.String())
		out = append(out, mergedMsg)
	}
	out = append(out, recent...)
	return trimChatMessages(out, compressTargetTokens)
}

// trimChatMessages 从最旧消息开始丢弃，直到总 token 不超过目标。
func trimChatMessages(messages []dto.Message, target int) []dto.Message {
	for len(messages) > 1 {
		total := 0
		for i := range messages {
			total += estimateTextTokens(messages[i].Role) + estimateTextTokens(messages[i].StringContent())
		}
		if total <= target {
			break
		}
		messages = messages[1:]
	}
	return messages
}

// CompressResponsesInput 对 responses 协议 input 做兜底压缩。
// input 支持 string 或 item 数组两种形态，压缩后统一输出为 item 数组。
func CompressResponsesInput(input json.RawMessage) json.RawMessage {
	items, err := parseResponsesInputItems(input)
	if err != nil || len(items) == 0 {
		return input
	}
	total := 0
	for i := range items {
		total += estimateResponsesItemTokens(items[i])
	}
	if total <= compressTriggerTokens {
		return input
	}

	// 从后往前定位第 keepRounds 个 user 消息，其后全部保留。
	keepStart := 0
	userCount := 0
	for i := len(items) - 1; i >= 0; i-- {
		if isResponsesUserMessage(items[i]) {
			userCount++
			if userCount == compressKeepRounds {
				keepStart = i
				break
			}
		}
	}
	recent := items[keepStart:]
	early := items[:keepStart]

	// 早期 item 中仅合并 message 文本；function_call/reasoning 等直接丢弃。
	var merged strings.Builder
	for i := range early {
		if text := responsesItemText(early[i]); text != "" {
			role := kitutil.Interface2String(early[i]["role"])
			if role == "" {
				role = "user"
			}
			merged.WriteString(role)
			merged.WriteString(": ")
			merged.WriteString(text)
			merged.WriteString("\n")
		}
	}

	out := make([]map[string]any, 0, len(recent)+1)
	if merged.Len() > 0 {
		out = append(out, responsesTextMessage("[以下为此前对话的合并历史]\n"+merged.String()))
	}
	out = append(out, recent...)
	for len(out) > 1 {
		total = 0
		for i := range out {
			total += estimateResponsesItemTokens(out[i])
		}
		if total <= compressTargetTokens {
			break
		}
		out = out[1:]
	}
	data, err := json.Marshal(out)
	if err != nil {
		return input
	}
	return data
}

// parseResponsesInputItems 解析 responses 协议 input（string 或数组形态）。
func parseResponsesInputItems(raw json.RawMessage) ([]map[string]any, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "null" {
		return nil, nil
	}
	if strings.HasPrefix(trimmed, `"`) {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return nil, err
		}
		return []map[string]any{{"role": "user", "content": text}}, nil
	}
	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, err
	}
	return items, nil
}

// isResponsesUserMessage 判断 responses item 是否为 user 消息。
func isResponsesUserMessage(item map[string]any) bool {
	return kitutil.Interface2String(item["type"]) == "message" && kitutil.Interface2String(item["role"]) == "user"
}

// responsesItemText 提取 responses message item 的文本内容。
func responsesItemText(item map[string]any) string {
	if kitutil.Interface2String(item["type"]) != "message" {
		return ""
	}
	switch content := item["content"].(type) {
	case string:
		return content
	case []any:
		var b strings.Builder
		for _, part := range content {
			partMap, ok := part.(map[string]any)
			if !ok {
				continue
			}
			partType := kitutil.Interface2String(partMap["type"])
			if partType == "input_text" || partType == "text" {
				b.WriteString(kitutil.Interface2String(partMap["text"]))
			}
		}
		return b.String()
	}
	return ""
}

// responsesTextMessage 构造一条纯文本 user 消息 item。
func responsesTextMessage(text string) map[string]any {
	return map[string]any{
		"type": "message",
		"role": "user",
		"content": []any{
			map[string]any{"type": "input_text", "text": text},
		},
	}
}

// estimateResponsesItemTokens 估算 responses item 的 token 数。
func estimateResponsesItemTokens(item map[string]any) int {
	tokens := estimateTextTokens(kitutil.Interface2String(item["role"]))
	tokens += estimateTextTokens(kitutil.Interface2String(item["type"]))
	tokens += estimateTextTokens(responsesItemText(item))
	return tokens
}
