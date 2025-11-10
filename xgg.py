import os
import time
import json
import hashlib
import hmac
import requests  # pip install requests


def send_hunyuan_3d_job(prompt: str):
    # Usa variables de entorno o pon tus claves directamente
    secret_id = 'IKID8LNDpaBcR6ZopcEdopqbPE70jGbmlZFj'
    secret_key = 'PcchhgfbNJ7lDUXMxMXUDi1FF2ThokTz'
    region = os.getenv("TENCENTCLOUD_REGION", "ap-guangzhou")

    service = "hunyuan"
    host = "hunyuan.intl.tencentcloudapi.com"
    endpoint = f"https://{host}"
    action = "SubmitHunyuanTo3DProJob"
    version = "2023-09-01"
    content_type = "application/json; charset=utf-8"

    # Solo lo esencial: el prompt
    payload = {
        "Prompt": prompt
    }

    # ---- TC3-HMAC-SHA256 signature ----
    timestamp = int(time.time())
    t = time.gmtime(timestamp)
    date = time.strftime("%Y-%m-%d", t)

    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""

    payload_json = json.dumps(
        payload, separators=(",", ":"), ensure_ascii=False)
    hashed_request_payload = hashlib.sha256(
        payload_json.encode("utf-8")).hexdigest()

    canonical_headers = f"content-type:{content_type}\nhost:{host}\n"
    signed_headers = "content-type;host"

    canonical_request = (
        f"{http_request_method}\n"
        f"{canonical_uri}\n"
        f"{canonical_querystring}\n"
        f"{canonical_headers}\n"
        f"{signed_headers}\n"
        f"{hashed_request_payload}"
    )

    hashed_canonical_request = hashlib.sha256(
        canonical_request.encode("utf-8")
    ).hexdigest()

    credential_scope = f"{date}/{service}/tc3_request"
    string_to_sign = (
        "TC3-HMAC-SHA256\n"
        f"{timestamp}\n"
        f"{credential_scope}\n"
        f"{hashed_canonical_request}"
    )

    def hmac_sha256(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    secret_date = hmac_sha256(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = hmac.new(secret_date, service.encode(
        "utf-8"), hashlib.sha256).digest()
    secret_signing = hmac.new(
        secret_service, b"tc3_request", hashlib.sha256).digest()
    signature = hmac.new(
        secret_signing,
        string_to_sign.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    authorization = (
        f"TC3-HMAC-SHA256 "
        f"Credential={secret_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    headers = {
        "Authorization": authorization,
        "Content-Type": content_type,
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Version": version,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Region": region,
    }

    # Enviamos la request. No mostramos el resultado del modelo.
    resp = requests.post(endpoint, headers=headers,
                         data=payload_json.encode("utf-8"))

    # Solo algo mínimo para saber si llegó (sin JobId ni ResultFile3Ds).
    if resp.status_code == 200:
        print("Solicitud enviada correctamente a Hunyuan3D Pro.")
    else:
        print("Error HTTP:", resp.status_code)
        print(resp.text)


if __name__ == "__main__":
    parrot_prompt = (
        "A detailed, colorful 3D parrot perched on a branch, "
        "realistic style, game-ready 3D model."
    )
    send_hunyuan_3d_job(parrot_prompt)
