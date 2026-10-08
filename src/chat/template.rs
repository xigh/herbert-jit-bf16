//! The Qwen3.5 chat template, rendered turn by turn.
//!
//! This is the model's own `chat_template.jinja`, written as functions: every
//! string below is byte for byte what the template produces. The session
//! never renders a whole conversation — the state is live — so each function
//! renders one piece: the system message (with the tools block when tools
//! are on), a user turn, the assistant header (thinking open or closed), and
//! the tool responses.

use crate::chat::tools::Tool;

/// The instructions the template prints after the tool list.
const TOOL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// The tools block alone: what an activation in the middle of a conversation
/// carries in a system message of its own.
pub fn tools_block(tools: &[Tool]) -> String {
    let mut s = String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for tool in tools {
        s.push('\n');
        s.push_str(&tool.to_json());
    }
    s.push_str("\n</tools>");
    s.push_str(TOOL_INSTRUCTIONS);
    s
}

/// `<|im_start|>system\n…<|im_end|>\n`: the tools block when there are tools,
/// then the system text. Nothing at all without either.
pub fn system_message(system: Option<&str>, tools: &[Tool]) -> String {
    let system = system.map(str::trim).filter(|s| !s.is_empty());
    match (system, tools.is_empty()) {
        (None, true) => String::new(),
        (Some(text), true) => format!("<|im_start|>system\n{text}<|im_end|>\n"),
        (None, false) => format!("<|im_start|>system\n{}<|im_end|>\n", tools_block(tools)),
        (Some(text), false) => format!("<|im_start|>system\n{}\n\n{text}<|im_end|>\n", tools_block(tools)),
    }
}

pub fn user_turn(text: &str) -> String {
    format!("<|im_start|>user\n{}<|im_end|>\n", text.trim())
}

/// The assistant header: the model then continues. With thinking, the
/// template opens `<think>` and the model writes its reasoning first; without
/// it, an empty thinking block is written for it.
pub fn assistant_header(think: bool) -> String {
    if think {
        "<|im_start|>assistant\n<think>\n".to_string()
    } else {
        "<|im_start|>assistant\n<think>\n\n</think>\n\n".to_string()
    }
}

/// The results of the tools the assistant called, as one user turn.
pub fn tool_responses(results: &[String]) -> String {
    let mut s = String::from("<|im_start|>user");
    for r in results {
        s.push_str("\n<tool_response>\n");
        s.push_str(r);
        s.push_str("\n</tool_response>");
    }
    s.push_str("<|im_end|>\n");
    s
}
