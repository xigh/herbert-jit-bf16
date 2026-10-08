//! Tools: what the model may call, how a call is written, how it is read back.
//!
//! A tool is a name, a description, a JSON schema of its parameters and the
//! function that runs it. The model calls it in the Qwen3.5 format:
//!
//! ```text
//! <tool_call>
//! <function=current_date_time>
//! <parameter=timezone>
//! Europe/Paris
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! The session hands the text between the two `<tool_call>` tokens to
//! [`parse_call`]; the handler's result goes back to the model as a
//! `<tool_response>`.

use std::collections::BTreeMap;

pub type Handler = Box<dyn Fn(&BTreeMap<String, String>) -> String + Send + Sync>;

pub struct Tool {
    pub name: String,
    pub description: String,
    /// A JSON schema, written as text so that it is printed exactly as given.
    pub parameters: String,
    pub handler: Handler,
}

impl Tool {
    /// The line printed in the `<tools>` block: the format the template's
    /// `tojson` produces (`", "` and `": "` separators, keys in this order).
    pub fn to_json(&self) -> String {
        format!(
            "{{\"type\": \"function\", \"function\": {{\"name\": {}, \"description\": {}, \"parameters\": {}}}}}",
            json_string(&self.name),
            json_string(&self.description),
            self.parameters
        )
    }
}

/// A JSON string literal, escaped as `json.dumps(ensure_ascii=False)` does.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The one tool of the demonstration: the local date and time.
pub fn current_date_time() -> Tool {
    Tool {
        name: "current_date_time".to_string(),
        description: "Returns the current local date and time of the machine, with the day of the week, in ISO 8601 form. Call it whenever the user asks what day or what time it is.".to_string(),
        parameters: "{\"properties\": {}, \"required\": [], \"type\": \"object\"}".to_string(),
        handler: Box::new(|_args| local_date_time()),
    }
}

/// `YYYY-MM-DD HH:MM:SS (weekday), local time`, from the C library: no
/// dependency for a clock.
fn local_date_time() -> String {
    let now = unsafe { libc::time(std::ptr::null_mut()) };
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    #[cfg(unix)]
    unsafe {
        libc::localtime_r(&now, &mut tm);
    }
    #[cfg(windows)]
    unsafe {
        libc::localtime_s(&mut tm, &now);
    }
    const DAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02} ({}), local time",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
        DAYS[(tm.tm_wday.rem_euclid(7)) as usize]
    )
}

/// A call the model wrote.
#[derive(Debug, PartialEq, Eq)]
pub struct Call {
    pub name: String,
    pub arguments: BTreeMap<String, String>,
}

/// Read `<function=name>` and its `<parameter=key>value</parameter>` blocks.
/// Values are trimmed of the newline that follows the opening tag and
/// precedes the closing one; what is inside is kept verbatim.
pub fn parse_call(text: &str) -> Option<Call> {
    let text = text.trim();
    let start = text.find("<function=")? + "<function=".len();
    let end = start + text[start..].find('>')?;
    let name = text[start..end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let body = &text[end + 1..];
    let body = body.strip_suffix("</function>").unwrap_or(body);
    let mut arguments = BTreeMap::new();
    let mut rest = body;
    while let Some(p) = rest.find("<parameter=") {
        let key_start = p + "<parameter=".len();
        let key_end = key_start + rest[key_start..].find('>')?;
        let key = rest[key_start..key_end].trim().to_string();
        let value_start = key_end + 1;
        let value_end = value_start + rest[value_start..].find("</parameter>")?;
        let value = rest[value_start..value_end].strip_prefix('\n').unwrap_or(&rest[value_start..value_end]);
        let value = value.strip_suffix('\n').unwrap_or(value);
        arguments.insert(key, value.to_string());
        rest = &rest[value_end + "</parameter>".len()..];
    }
    Some(Call { name, arguments })
}
