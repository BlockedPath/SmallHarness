use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::auth::{auth_file_path, AuthStore, OAuthCredential};
use crate::input::plain_read_line;

pub const PROVIDER: &str = "xai-oauth";
const LEGACY_PROVIDER: &str = "xai-auth";
const DISCOVERY_URL: &str = "https://auth.x.ai/.well-known/openid-configuration";
const CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
const REDIRECT_HOST: &str = "127.0.0.1";
const PREFERRED_REDIRECT_PORT: u16 = 56121;
const REDIRECT_PATH: &str = "/callback";
const TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
const REFRESH_SKEW_SECS: u64 = 120;

const GROK_CLI_AUTH_SCOPE_KEY: &str = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828";
const GROK_CLI_LEGACY_AUTH_SCOPE_KEY: &str = "https://accounts.x.ai/sign-in";

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn random_bytes<const N: usize>() -> [u8; N] {
    let mut out = [0u8; N];
    if let Ok(mut file) = fs::File::open("/dev/urandom") {
        if file.read_exact(&mut out).is_ok() {
            return out;
        }
    }
    let seed = format!(
        "{}:{}:{}",
        now_secs(),
        std::process::id(),
        std::thread::current().name().unwrap_or("small-harness")
    );
    let mut digest = Sha256::digest(seed.as_bytes()).to_vec();
    while digest.len() < N {
        let next = Sha256::digest(&digest);
        digest.extend_from_slice(&next);
    }
    out.copy_from_slice(&digest[..N]);
    out
}

fn random_hex(bytes: usize) -> String {
    random_bytes::<32>()[..bytes]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn pkce_pair() -> (String, String) {
    let verifier = URL_SAFE_NO_PAD.encode(random_bytes::<32>());
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    (verifier, challenge)
}

fn percent_encode(input: &str) -> String {
    let mut out = String::new();
    for b in input.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn percent_decode(input: &str) -> String {
    let mut out = Vec::new();
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(value) = u8::from_str_radix(hex, 16) {
                    out.push(value);
                    i += 3;
                    continue;
                }
            }
        }
        out.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn form_urlencoded(params: &[(&str, &str)]) -> String {
    params
        .iter()
        .map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

fn query_param(path: &str, name: &str) -> Option<String> {
    let raw_query = path.split_once('?').map(|(_, q)| q).unwrap_or(path);
    let query = raw_query
        .split_once('#')
        .map(|(q, _)| q)
        .unwrap_or(raw_query);
    for pair in query.split('&') {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        if k == name {
            return Some(percent_decode(v));
        }
    }
    None
}

fn parse_authorization_input(input: &str) -> (Option<String>, Option<String>) {
    let value = input.trim();
    if value.contains("code=") || value.contains("state=") {
        (query_param(value, "code"), query_param(value, "state"))
    } else if value.is_empty() {
        (None, None)
    } else {
        (Some(value.to_string()), None)
    }
}

fn parse_expiry_secs(value: &Value) -> Option<u64> {
    if let Some(n) = value.as_u64() {
        return Some(if n > 100_000_000_000 { n / 1000 } else { n });
    }
    let s = value.as_str()?.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(n) = s.parse::<u64>() {
        return Some(if n > 100_000_000_000 { n / 1000 } else { n });
    }
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .and_then(|dt| u64::try_from(dt.timestamp()).ok())
}

fn grok_auth_file_path() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|home| PathBuf::from(home).join(".grok").join("auth.json"))
}

fn oauth_from_grok_cli_json(data: &Value) -> Option<OAuthCredential> {
    if let Some(oidc) = data.get(GROK_CLI_AUTH_SCOPE_KEY).and_then(Value::as_object) {
        let access = oidc
            .get("key")
            .or_else(|| oidc.get("access_token"))
            .or_else(|| oidc.get("token"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !access.is_empty() {
            let expires = oidc
                .get("expires_at")
                .and_then(parse_expiry_secs)
                .unwrap_or_else(|| now_secs() + 6 * 60 * 60)
                .saturating_sub(REFRESH_SKEW_SECS);
            return Some(OAuthCredential {
                credential_type: "oauth".into(),
                access: access.to_string(),
                refresh: oidc
                    .get("refresh_token")
                    .or_else(|| oidc.get("refresh"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                expires,
                account_id: None,
            });
        }
    }

    let legacy_access = data
        .get(GROK_CLI_LEGACY_AUTH_SCOPE_KEY)
        .and_then(Value::as_object)
        .and_then(|legacy| {
            legacy
                .get("key")
                .or_else(|| legacy.get("access_token"))
                .or_else(|| legacy.get("token"))
                .and_then(Value::as_str)
        });
    if let Some(access) = legacy_access.filter(|s| !s.is_empty()) {
        return Some(OAuthCredential {
            credential_type: "oauth".into(),
            access: access.to_string(),
            refresh: String::new(),
            expires: now_secs() + 30 * 24 * 60 * 60,
            account_id: None,
        });
    }

    let access = data
        .get("access_token")
        .or_else(|| data.get("token"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if access.is_empty() {
        return None;
    }
    Some(OAuthCredential {
        credential_type: "oauth".into(),
        access: access.to_string(),
        refresh: data
            .get("refresh_token")
            .or_else(|| data.get("refresh"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        expires: data
            .get("expires_at")
            .or_else(|| data.get("expires"))
            .and_then(parse_expiry_secs)
            .unwrap_or_else(|| now_secs() + 30 * 24 * 60 * 60),
        account_id: None,
    })
}

pub fn grok_cli_oauth_credentials() -> Option<OAuthCredential> {
    let path = grok_auth_file_path()?;
    let text = fs::read_to_string(path).ok()?;
    let json: Value = serde_json::from_str(&text).ok()?;
    oauth_from_grok_cli_json(&json)
}

pub fn has_oauth_credentials() -> bool {
    let store = AuthStore::load();
    store.get_oauth(PROVIDER).is_some()
        || store.get_oauth(LEGACY_PROVIDER).is_some()
        || grok_cli_oauth_credentials().is_some()
}

#[derive(Debug, Deserialize)]
struct DiscoveryResponse {
    authorization_endpoint: String,
    token_endpoint: String,
}

fn validate_xai_endpoint(url: &str) -> Result<String> {
    let parsed = reqwest::Url::parse(url)?;
    let host = parsed.host_str().unwrap_or_default().to_lowercase();
    if parsed.scheme() != "https" || (host != "x.ai" && !host.ends_with(".x.ai")) {
        return Err(anyhow!(
            "xAI OAuth discovery returned unexpected endpoint: {url}"
        ));
    }
    Ok(url.to_string())
}

async fn discovery(client: &reqwest::Client) -> Result<DiscoveryResponse> {
    let resp = client
        .get(DISCOVERY_URL)
        .header("accept", "application/json")
        .send()
        .await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!(
            "xAI OAuth discovery failed ({status}): {}",
            body.trim()
        ));
    }
    let raw: DiscoveryResponse = resp.json().await?;
    Ok(DiscoveryResponse {
        authorization_endpoint: validate_xai_endpoint(&raw.authorization_endpoint)?,
        token_endpoint: validate_xai_endpoint(&raw.token_endpoint)?,
    })
}

fn authorization_url(
    authorization_endpoint: &str,
    redirect_uri: &str,
    challenge: &str,
    state: &str,
    nonce: &str,
) -> String {
    let params = [
        ("response_type", "code"),
        ("client_id", CLIENT_ID),
        ("redirect_uri", redirect_uri),
        ("scope", SCOPE),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
        ("nonce", nonce),
    ];
    format!("{authorization_endpoint}?{}", form_urlencoded(&params))
}

fn callback_cors_origin(origin: Option<&str>) -> Option<&str> {
    match origin {
        Some("https://accounts.x.ai") | Some("https://auth.x.ai") => origin,
        _ => None,
    }
}

fn request_header<'a>(request: &'a str, name: &str) -> Option<&'a str> {
    request.lines().skip(1).find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.eq_ignore_ascii_case(name).then(|| value.trim())
    })
}

fn write_raw_response(
    mut stream: TcpStream,
    status: &str,
    content_type: &str,
    body: &str,
    cors_origin: Option<&str>,
) {
    let mut response = format!("HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\n");
    if let Some(origin) = cors_origin {
        response.push_str(&format!(
            "Access-Control-Allow-Origin: {origin}\r\nAccess-Control-Allow-Methods: GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nAccess-Control-Allow-Private-Network: true\r\nVary: Origin\r\n"
        ));
    }
    response.push_str(&format!(
        "content-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    ));
    let _ = stream.write_all(response.as_bytes());
}

fn write_callback_response(stream: TcpStream, ok: bool, message: &str, cors_origin: Option<&str>) {
    let title = if ok {
        "xAI authorization received"
    } else {
        "xAI authorization failed"
    };
    let body = format!(
        "<!doctype html><meta charset=\"utf-8\"><title>{}</title><body style=\"font-family: system-ui; margin: 3rem\"><h1>{}</h1><p>{}</p></body>",
        title, title, message
    );
    let status = if ok { "200 OK" } else { "400 Bad Request" };
    write_raw_response(
        stream,
        status,
        "text/html; charset=utf-8",
        &body,
        cors_origin,
    );
}

fn bind_callback_listener() -> Result<(TcpListener, String)> {
    let listener = TcpListener::bind((REDIRECT_HOST, PREFERRED_REDIRECT_PORT))
        .or_else(|_| TcpListener::bind((REDIRECT_HOST, 0)))
        .context("binding xAI OAuth callback server")?;
    let port = listener.local_addr()?.port();
    Ok((
        listener,
        format!("http://{REDIRECT_HOST}:{port}{REDIRECT_PATH}"),
    ))
}

fn wait_for_browser_callback(listener: TcpListener, state: String) -> Result<String> {
    loop {
        let (mut stream, _) = listener
            .accept()
            .context("waiting for xAI OAuth callback")?;
        let mut buf = [0u8; 8192];
        let n = stream.read(&mut buf).unwrap_or(0);
        let request = String::from_utf8_lossy(&buf[..n]);
        let first = request.lines().next().unwrap_or_default();
        let method = first.split_whitespace().next().unwrap_or_default();
        let path = first.split_whitespace().nth(1).unwrap_or_default();
        let origin_owned = request_header(&request, "Origin").map(ToString::to_string);
        let cors_origin = callback_cors_origin(origin_owned.as_deref());

        if method == "OPTIONS" {
            write_raw_response(
                stream,
                "204 No Content",
                "text/plain; charset=utf-8",
                "",
                cors_origin,
            );
            continue;
        }
        if !path.starts_with(REDIRECT_PATH) {
            write_raw_response(
                stream,
                "404 Not Found",
                "text/plain; charset=utf-8",
                "Not found",
                cors_origin,
            );
            continue;
        }
        if let Some(error) = query_param(path, "error") {
            let desc = query_param(path, "error_description").unwrap_or_else(|| error.clone());
            write_callback_response(stream, false, &desc, cors_origin);
            return Err(anyhow!("xAI authorization failed: {desc}"));
        }
        let got_state = query_param(path, "state");
        if got_state.as_deref() != Some(state.as_str()) {
            write_callback_response(stream, false, "State mismatch.", cors_origin);
            return Err(anyhow!("xAI OAuth state mismatch"));
        }
        let code = query_param(path, "code").ok_or_else(|| anyhow!("xAI callback missing code"))?;
        write_callback_response(
            stream,
            true,
            "xAI authentication completed. You can close this window.",
            cors_origin,
        );
        return Ok(code);
    }
}

fn open_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    let mut cmd = std::process::Command::new("open");
    #[cfg(target_os = "linux")]
    let mut cmd = std::process::Command::new("xdg-open");
    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "start", ""]);
        c
    };
    cmd.arg(url);
    cmd.spawn().context("opening browser")?;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
}

async fn read_token_response(
    resp: reqwest::Response,
    operation: &str,
    fallback_refresh: &str,
) -> Result<OAuthCredential> {
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!(
            "xAI token {operation} failed ({status}): {}",
            body.trim()
        ));
    }
    let token: TokenResponse = resp.json().await?;
    let refresh = token
        .refresh_token
        .unwrap_or_else(|| fallback_refresh.to_string());
    if refresh.is_empty() {
        return Err(anyhow!(
            "xAI token response did not include a refresh token"
        ));
    }
    Ok(OAuthCredential {
        credential_type: "oauth".into(),
        access: token.access_token,
        refresh,
        expires: now_secs() + token.expires_in.unwrap_or(3600)
            - REFRESH_SKEW_SECS.min(token.expires_in.unwrap_or(3600)),
        account_id: None,
    })
}

async fn exchange_authorization_code(
    client: &reqwest::Client,
    token_endpoint: &str,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<OAuthCredential> {
    let body = form_urlencoded(&[
        ("grant_type", "authorization_code"),
        ("code", code),
        ("redirect_uri", redirect_uri),
        ("client_id", CLIENT_ID),
        ("code_verifier", verifier),
    ]);
    let resp = client
        .post(token_endpoint)
        .header("accept", "application/json")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await?;
    read_token_response(resp, "exchange", "").await
}

pub async fn refresh_oauth(client: &reqwest::Client, refresh: &str) -> Result<OAuthCredential> {
    let body = form_urlencoded(&[
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh),
        ("client_id", CLIENT_ID),
    ]);
    let resp = client
        .post(TOKEN_URL)
        .header("accept", "application/json")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await?;
    read_token_response(resp, "refresh", refresh).await
}

fn save_oauth(credential: OAuthCredential) -> Result<PathBuf> {
    let mut store = AuthStore::load();
    store.set_oauth(PROVIDER, credential);
    store.save()?;
    auth_file_path().context("no auth file path")
}

pub async fn login_browser(client: &reqwest::Client) -> Result<OAuthCredential> {
    let discovery = discovery(client).await?;
    let (listener, redirect_uri) = bind_callback_listener()?;
    let (verifier, challenge) = pkce_pair();
    let state = random_hex(16);
    let nonce = random_hex(16);
    let url = authorization_url(
        &discovery.authorization_endpoint,
        &redirect_uri,
        &challenge,
        &state,
        &nonce,
    );
    println!("  Open this URL to sign in with xAI/Grok:\n\n  {url}\n");
    if let Err(e) = open_browser(&url) {
        println!("  Browser did not open automatically: {e}");
    }
    println!("  Waiting for callback on {redirect_uri} ...");
    let state_for_thread = state.clone();
    let callback =
        tokio::task::spawn_blocking(move || wait_for_browser_callback(listener, state_for_thread))
            .await
            .context("joining xAI OAuth callback task")?;
    let code = match callback {
        Ok(code) => code,
        Err(e) => {
            println!("  Callback failed: {e}");
            let input = plain_read_line(
                "  Paste the authorization code or full redirect URL (blank to cancel): ".into(),
            )
            .await?;
            let (code, got_state) = parse_authorization_input(&input);
            if let Some(got_state) = got_state {
                if got_state != state {
                    return Err(anyhow!("xAI OAuth state mismatch"));
                }
            }
            code.ok_or_else(|| anyhow!("missing xAI authorization code"))?
        }
    };
    exchange_authorization_code(
        client,
        &discovery.token_endpoint,
        &code,
        &verifier,
        &redirect_uri,
    )
    .await
}

pub async fn login_and_save_browser(client: &reqwest::Client) -> Result<PathBuf> {
    if let Some(credential) = grok_cli_oauth_credentials() {
        let pick = plain_read_line(
            "  Found official Grok CLI credentials in ~/.grok/auth.json. Use them? [Y/n]: ".into(),
        )
        .await?;
        if pick.trim().is_empty() || pick.trim().to_lowercase().starts_with('y') {
            return save_oauth(credential);
        }
    }
    let credential = login_browser(client).await?;
    save_oauth(credential)
}

pub async fn access_token(client: &reqwest::Client) -> Result<String> {
    let store = AuthStore::load();
    let credential = store
        .get_oauth(PROVIDER)
        .or_else(|| store.get_oauth(LEGACY_PROVIDER))
        .cloned()
        .or_else(grok_cli_oauth_credentials)
        .ok_or_else(|| {
            anyhow!("not logged in for xAI/Grok; run `/login xai` or set XAI_API_KEY")
        })?;

    if credential.expires <= now_secs() + 60 && !credential.refresh.is_empty() {
        let refreshed = refresh_oauth(client, &credential.refresh).await?;
        let mut store = AuthStore::load();
        store.set_oauth(PROVIDER, refreshed.clone());
        store.save()?;
        Ok(refreshed.access)
    } else {
        Ok(credential.access)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorization_url_has_xai_oauth_params() {
        let url = authorization_url(
            "https://auth.x.ai/oauth2/auth",
            "http://127.0.0.1:1/callback",
            "challenge",
            "state",
            "nonce",
        );
        assert!(url.starts_with("https://auth.x.ai"));
        assert!(url.contains("client_id=b1a00492-073a-47ea-816f-4c329264a828"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("grok-cli%3Aaccess"));
    }

    #[test]
    fn parses_redirect_url_input() {
        let (code, state) =
            parse_authorization_input("http://127.0.0.1:56121/callback?code=abc%20123&state=st");
        assert_eq!(code.as_deref(), Some("abc 123"));
        assert_eq!(state.as_deref(), Some("st"));
    }

    #[test]
    fn reads_grok_cli_oidc_shape() {
        let data = serde_json::json!({
            GROK_CLI_AUTH_SCOPE_KEY: {
                "key": "access",
                "refresh_token": "refresh",
                "expires_at": 4_000_000_000u64,
            }
        });
        let credential = oauth_from_grok_cli_json(&data).unwrap();
        assert_eq!(credential.access, "access");
        assert_eq!(credential.refresh, "refresh");
        assert!(credential.expires > 0);
    }
}
