//! Pairing: a computer gets its certificate with a short code instead of a file handed over.
//!
//! 1. On the server, `lucem serve` (or `lucem clients pair`) makes a code such as `42-8317-2956`:
//!    the first part names it (public), the rest is the secret. It works once, for
//!    [`CODE_MINUTES`] minutes and [`CODE_TRIES`] tries. Codes live in the authority's folder
//!    (`pairing.json`, mode 0600), so a running server sees new ones at once.
//! 2. The computer connects (encrypted, the server not yet trusted) and both sides run SPAKE2
//!    with the secret: each learns a shared key only if both know the code, and the code itself
//!    never crosses the network. A man in the middle gets one guess per try, then the code is
//!    spent.
//! 3. The computer makes its own key pair and sends only the public key, with a MAC under the
//!    shared key (it knows the code). The server issues a certificate for it, records the client
//!    in its registry, and answers with the certificate and its CA, MAC'd the same way (it knows
//!    the code: the CA is the server's own).
//!
//! From then on the computer connects with mutual TLS as any enrolled client; revoking works
//! the same.

use std::path::{Path, PathBuf};

use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use spake2::{Ed25519Group, Identity, Password, Spake2};

use crate::auth::{now_unix, Authority, ClientIdentity, Permission};
use crate::error::{StreamError, StreamResult};

/// How long a code works (minutes).
pub const CODE_MINUTES: u64 = 10;
/// Wrong tries before a code is spent.
pub const CODE_TRIES: u32 = 3;

const CODES_FILE: &str = "pairing.json";
const ID_CLIENT: &[u8] = b"lucem-pair-client";
const ID_SERVER: &[u8] = b"lucem-pair-server";

/// A pairing code: `id-secret` as people type it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairingCode {
    /// Two digits naming the code (sent in the clear).
    pub id: String,
    /// Eight digits, the shared secret (never sent).
    pub secret: String,
    pub permission: Permission,
    pub expires_unix: u64,
    pub tries_left: u32,
}

impl PairingCode {
    /// `42-8317-2956`.
    pub fn display(&self) -> String {
        format!("{}-{}-{}", self.id, &self.secret[..4], &self.secret[4..])
    }
}

/// A live code as listed: everything but its secret.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveCode {
    pub id: String,
    pub permission: Permission,
    pub expires_unix: u64,
    pub tries_left: u32,
}

/// `42-8317-2956` (spaces and dashes optional) as its id and secret.
pub fn parse_code(code: &str) -> Option<(String, String)> {
    let digits: String = code.chars().filter(|c| !c.is_whitespace() && *c != '-').collect();
    (digits.len() == 10 && digits.chars().all(|c| c.is_ascii_digit())).then(|| (digits[..2].to_string(), digits[2..].to_string()))
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Codes {
    codes: Vec<PairingCode>,
}

fn random_digits(n: usize) -> StreamResult<String> {
    let mut bytes = vec![0u8; n];
    getrandom::fill(&mut bytes).map_err(|e| StreamError::Tls(format!("no randomness: {e}")))?;
    // 250 = 25 × 10: values above it are drawn again, so every digit is equally likely
    let mut out = String::with_capacity(n);
    let mut pool = bytes.into_iter();
    while out.len() < n {
        let b = match pool.next() {
            Some(b) => b,
            None => {
                let mut one = [0u8; 1];
                getrandom::fill(&mut one).map_err(|e| StreamError::Tls(format!("no randomness: {e}")))?;
                one[0]
            }
        };
        if b < 250 {
            out.push(char::from(b'0' + b % 10));
        }
    }
    Ok(out)
}

impl Authority {
    fn codes_path(&self) -> PathBuf {
        self.dir().join(CODES_FILE)
    }

    fn read_codes(&self) -> StreamResult<Codes> {
        match std::fs::read_to_string(self.codes_path()) {
            Ok(t) => Ok(serde_json::from_str(&t)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Codes::default()),
            Err(e) => Err(e.into()),
        }
    }

    fn write_codes(&self, codes: &Codes) -> StreamResult<()> {
        crate::auth::replace_private(&self.codes_path(), serde_json::to_string_pretty(codes)?.as_bytes())
    }

    /// A new pairing code for a computer with `permission`, valid [`CODE_MINUTES`] minutes.
    pub fn new_pairing_code(&self, permission: Permission) -> StreamResult<PairingCode> {
        let mut codes = self.read_codes()?;
        let now = now_unix();
        codes.codes.retain(|c| c.expires_unix > now && c.tries_left > 0);
        let id = loop {
            let id = random_digits(2)?;
            if !codes.codes.iter().any(|c| c.id == id) {
                break id;
            }
        };
        let code = PairingCode { id, secret: random_digits(8)?, permission, expires_unix: now + CODE_MINUTES * 60, tries_left: CODE_TRIES };
        codes.codes.push(code.clone());
        self.write_codes(&codes)?;
        Ok(code)
    }

    /// The live code named `id`, one try taken from it (a try counts before the answer is known,
    /// so guessing in parallel gains nothing).
    fn take_try(&self, id: &str) -> StreamResult<PairingCode> {
        let mut codes = self.read_codes()?;
        let now = now_unix();
        codes.codes.retain(|c| c.expires_unix > now && c.tries_left > 0);
        let Some(code) = codes.codes.iter_mut().find(|c| c.id == id) else {
            self.write_codes(&codes)?;
            return Err(StreamError::Server("no such pairing code, or it has expired: make a new one on the server".into()));
        };
        code.tries_left -= 1;
        let taken = code.clone();
        self.write_codes(&codes)?;
        Ok(taken)
    }

    /// The live codes, without their secrets: what an operator may see to decide what to cancel.
    pub fn pairing_codes(&self) -> StreamResult<Vec<LiveCode>> {
        let now = now_unix();
        Ok(self.read_codes()?.codes.into_iter().filter(|c| c.expires_unix > now && c.tries_left > 0).map(|c| LiveCode { id: c.id, permission: c.permission, expires_unix: c.expires_unix, tries_left: c.tries_left }).collect())
    }

    /// Cancels the live code named `id`, or every code (`None`): a code that may have been seen
    /// stops working at once instead of at its expiry. Returns how many were cancelled.
    pub fn cancel_pairing_codes(&self, id: Option<&str>) -> StreamResult<usize> {
        let mut codes = self.read_codes()?;
        let now = now_unix();
        codes.codes.retain(|c| c.expires_unix > now && c.tries_left > 0);
        let before = codes.codes.len();
        match id {
            Some(id) => codes.codes.retain(|c| c.id != id),
            None => codes.codes.clear(),
        }
        let cancelled = before - codes.codes.len();
        self.write_codes(&codes)?;
        Ok(cancelled)
    }

    /// Spends the code named `id` (it paired a computer).
    fn spend(&self, id: &str) -> StreamResult<()> {
        let mut codes = self.read_codes()?;
        codes.codes.retain(|c| c.id != id);
        self.write_codes(&codes)
    }
}

// ---- the exchange ------------------------------------------------------------------------------

/// The computer's first message: which code, its name, its SPAKE2 message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairHello {
    pub id: String,
    pub name: String,
    pub spake: Vec<u8>,
}

/// The server's SPAKE2 message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "pair", rename_all = "snake_case", deny_unknown_fields)]
pub enum PairReply {
    Spake { spake: Vec<u8> },
    /// The certificate issued, the CA, the server's names (to connect by) and the permission,
    /// with a MAC proving the server knows the code.
    Paired { certificate_pem: String, ca_pem: String, names: Vec<String>, permission: Permission, mac: Vec<u8> },
    Error { reason: String },
}

/// The computer's public key (SubjectPublicKeyInfo, DER) and its MAC.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairKey {
    pub public_key: Vec<u8>,
    pub mac: Vec<u8>,
}

fn mac(key: &[u8], parts: &[&[u8]]) -> Vec<u8> {
    let mut m = <Hmac<Sha256> as KeyInit>::new_from_slice(key).expect("HMAC takes any key length");
    for p in parts {
        m.update(&(p.len() as u64).to_le_bytes());
        m.update(p);
    }
    m.finalize().into_bytes().to_vec()
}

fn mac_ok(key: &[u8], parts: &[&[u8]], tag: &[u8]) -> bool {
    let mut m = <Hmac<Sha256> as KeyInit>::new_from_slice(key).expect("HMAC takes any key length");
    for p in parts {
        m.update(&(p.len() as u64).to_le_bytes());
        m.update(p);
    }
    m.verify_slice(tag).is_ok()
}

/// The server's side, on one stream: `hello` read, the rest exchanged.
pub(crate) async fn serve_pairing(authority: &Authority, names: &[String], hello: PairHello, send: &mut quinn::SendStream, recv: &mut quinn::RecvStream) -> StreamResult<()> {
    use crate::protocol::{read_message, write_message};
    let code = match authority.take_try(&hello.id) {
        Ok(c) => c,
        Err(e) => return write_message(send, &PairReply::Error { reason: e.to_string() }).await,
    };
    if crate::auth::check_name(&hello.name).is_err() {
        return write_message(send, &PairReply::Error { reason: "a computer's name is letters, digits, '-', '_' or '.'".into() }).await;
    }
    let (state, outbound) = Spake2::<Ed25519Group>::start_b(&Password::new(code.secret.as_bytes()), &Identity::new(ID_CLIENT), &Identity::new(ID_SERVER));
    write_message(send, &PairReply::Spake { spake: outbound }).await?;
    let key = match state.finish(&hello.spake) {
        Ok(k) => k,
        Err(_) => return write_message(send, &PairReply::Error { reason: "pairing failed".into() }).await,
    };
    let pk: PairKey = read_message(recv).await?;
    if !mac_ok(&key, &[b"client", hello.name.as_bytes(), &pk.public_key], &pk.mac) {
        tracing::warn!("pairing as {:?} failed: wrong code ({} tries left)", hello.name, code.tries_left);
        return write_message(send, &PairReply::Error { reason: "wrong code".into() }).await;
    }
    let (record, certificate_pem) = match authority.enroll_public_key(&hello.name, code.permission, &pk.public_key) {
        Ok(r) => r,
        Err(e) => return write_message(send, &PairReply::Error { reason: e.to_string() }).await,
    };
    authority.spend(&code.id)?;
    let ca_pem = authority.ca_pem().to_string();
    let names_joined = names.join(",");
    let tag = mac(&key, &[b"server", certificate_pem.as_bytes(), ca_pem.as_bytes(), names_joined.as_bytes()]);
    tracing::info!("paired {} ({})", record.name, record.permission);
    write_message(send, &PairReply::Paired { certificate_pem, ca_pem, names: names.to_vec(), permission: code.permission, mac: tag }).await
}

/// What a computer keeps after pairing.
#[derive(Debug, Clone)]
pub struct Paired {
    /// Certificate, private key and CA in one PEM (as an enrollment file): [`ClientIdentity::from_pem`].
    pub identity_pem: String,
    /// The names the server's certificate carries (connect with one of them).
    pub names: Vec<String>,
    pub permission: Permission,
    /// SHA-256 of the server's CA certificate (hex): how the server is recognised later.
    pub ca_fingerprint: String,
}

/// The computer's side: pairs with the server at `addr` using `code`, as `name`. Its private key
/// is made here and never sent.
pub async fn pair(addr: std::net::SocketAddr, code: &str, name: &str) -> StreamResult<Paired> {
    use crate::protocol::{read_message, write_message};
    let (id, secret) = parse_code(code).ok_or_else(|| StreamError::Server("a pairing code is ten digits, like 42-8317-2956".into()))?;
    // Encrypted, the server not yet trusted: SPAKE2 decides who is who
    // (accepting any certificate is safe here only because SPAKE2 then authenticates the server)
    let config = crate::tls::client_config_with(&crate::tls::ServerTrust::DangerAcceptAnyCertificate, None)?;
    let mut endpoint = quinn::Endpoint::client((if addr.is_ipv6() { "[::]:0" } else { "0.0.0.0:0" }).parse().expect("an address"))?;
    endpoint.set_default_client_config(config);
    let connection = endpoint.connect(addr, "lucem-pairing").map_err(StreamError::protocol)?.await.map_err(StreamError::protocol)?;
    let (mut send, mut recv) = connection.open_bi().await.map_err(StreamError::protocol)?;

    let (state, outbound) = Spake2::<Ed25519Group>::start_a(&Password::new(secret.as_bytes()), &Identity::new(ID_CLIENT), &Identity::new(ID_SERVER));
    write_message(&mut send, &crate::protocol::Request::Pair { hello: PairHello { id, name: name.into(), spake: outbound } }).await?;
    let spake = match read_message::<PairReply>(&mut recv).await? {
        PairReply::Spake { spake } => spake,
        PairReply::Error { reason } => return Err(StreamError::Server(reason)),
        PairReply::Paired { .. } => return Err(StreamError::Server("the server answered out of turn".into())),
    };
    let key = state.finish(&spake).map_err(|_| StreamError::Server("pairing failed".into()))?;
    let key_pair = rcgen::KeyPair::generate().map_err(|e| StreamError::Tls(e.to_string()))?;
    let public_key = rcgen::PublicKeyData::subject_public_key_info(&key_pair);
    let tag = mac(&key, &[b"client", name.as_bytes(), &public_key]);
    write_message(&mut send, &PairKey { public_key, mac: tag }).await?;
    let reply = read_message::<PairReply>(&mut recv).await?;
    connection.close(quinn::VarInt::from_u32(0), b"paired");
    match reply {
        PairReply::Paired { certificate_pem, ca_pem, names, permission, mac: tag } => {
            if !mac_ok(&key, &[b"server", certificate_pem.as_bytes(), ca_pem.as_bytes(), names.join(",").as_bytes()], &tag) {
                return Err(StreamError::Server("the server could not prove it knows the code: not paired".into()));
            }
            let identity_pem = format!("{certificate_pem}{}{ca_pem}", key_pair.serialize_pem());
            let identity = ClientIdentity::from_pem(identity_pem.as_bytes())?;
            Ok(Paired { ca_fingerprint: identity.ca_fingerprint(), identity_pem, names, permission })
        }
        PairReply::Error { reason } => Err(StreamError::Server(reason)),
        PairReply::Spake { .. } => Err(StreamError::Server("the server answered out of turn".into())),
    }
}

/// Writes `pem` readable by its owner only.
pub fn save_identity(path: &Path, pem: &str) -> StreamResult<()> {
    crate::auth::replace_private(path, pem.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_read_with_or_without_dashes() {
        assert_eq!(parse_code("42-8317-2956"), Some(("42".into(), "83172956".into())));
        assert_eq!(parse_code("42 8317 2956"), Some(("42".into(), "83172956".into())));
        assert_eq!(parse_code("42-8317-295"), None);
        assert_eq!(parse_code("ab-8317-2956"), None);
    }

    #[test]
    fn a_code_is_spent_after_its_tries() {
        let dir = std::env::temp_dir().join(format!("lucem-pair-codes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let authority = Authority::open_or_create(&dir).unwrap();
        let code = authority.new_pairing_code(Permission::Read).unwrap();
        assert_eq!(code.display().len(), 12);
        for _ in 0..CODE_TRIES {
            authority.take_try(&code.id).unwrap();
        }
        assert!(authority.take_try(&code.id).is_err(), "no tries left");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn live_codes_list_without_secrets_and_cancel() {
        let dir = std::env::temp_dir().join(format!("lucem-pair-cancel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let authority = Authority::open_or_create(&dir).unwrap();
        let a = authority.new_pairing_code(Permission::Read).unwrap();
        let b = authority.new_pairing_code(Permission::Control).unwrap();
        let listed = authority.pairing_codes().unwrap();
        assert_eq!(listed.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(), [a.id.as_str(), b.id.as_str()]);
        assert_eq!(listed[1].permission, Permission::Control);
        assert_eq!(authority.cancel_pairing_codes(Some(&a.id)).unwrap(), 1);
        assert!(authority.take_try(&a.id).is_err(), "a cancelled code no longer works");
        assert_eq!(authority.cancel_pairing_codes(Some("no such id")).unwrap(), 0);
        assert_eq!(authority.cancel_pairing_codes(None).unwrap(), 1);
        assert!(authority.pairing_codes().unwrap().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
