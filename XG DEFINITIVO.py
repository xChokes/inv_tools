import os
import time
import json
import hashlib
import hmac
import requests  # pip install requests


SECRET_ID = 'IKID8LNDpaBcR6ZopcEdopqbPE70jGbmlZFj'
SECRET_KEY = 'PcchhgfbNJ7lDUXMxMXUDi1FF2ThokTz'
REGION = "ap-singapore"

SERVICE = "hunyuan"
HOST = "hunyuan.intl.tencentcloudapi.com"
ENDPOINT = f"https://{HOST}"
VERSION = "2023-09-01"
CONTENT_TYPE = "application/json; charset=utf-8"


def sign_tc3(action: str, payload: dict, timestamp: int):
    algorithm = "TC3-HMAC-SHA256"
    t = time.gmtime(timestamp)
    date = time.strftime("%Y-%m-%d", t)

    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""

    payload_str = json.dumps(
        payload, separators=(",", ":"), ensure_ascii=False)
    hashed_payload = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    canonical_headers = f"content-type:{CONTENT_TYPE}\nhost:{HOST}\n"
    signed_headers = "content-type;host"

    canonical_request = (
        f"{http_request_method}\n"
        f"{canonical_uri}\n"
        f"{canonical_querystring}\n"
        f"{canonical_headers}\n"
        f"{signed_headers}\n"
        f"{hashed_payload}"
    )

    hashed_canonical_request = hashlib.sha256(
        canonical_request.encode("utf-8")
    ).hexdigest()

    credential_scope = f"{date}/{SERVICE}/tc3_request"
    string_to_sign = (
        f"{algorithm}\n"
        f"{timestamp}\n"
        f"{credential_scope}\n"
        f"{hashed_canonical_request}"
    )

    def sign(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    secret_date = sign(("TC3" + SECRET_KEY).encode("utf-8"), date)
    secret_service = sign(secret_date, SERVICE)
    secret_signing = sign(secret_service, "tc3_request")
    signature = hmac.new(
        secret_signing,
        string_to_sign.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    authorization = (
        f"{algorithm} "
        f"Credential={SECRET_ID}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    headers = {
        "Authorization": authorization,
        "Content-Type": CONTENT_TYPE,
        "Host": HOST,
        "X-TC-Action": action,
        "X-TC-Version": VERSION,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Region": REGION,
    }

    return headers, payload_str


def call_hunyuan_api(action: str, payload: dict) -> dict:
    timestamp = int(time.time())
    headers, body = sign_tc3(action, payload, timestamp)
    resp = requests.post(ENDPOINT, headers=headers, data=body.encode("utf-8"))

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    if "Response" not in data:
        raise RuntimeError(f"Respuesta inesperada: {data}")
    return data["Response"]


def submit_parrot_job() -> str:
    prompt = (
        "A detailed, colorful 3D parrot perched on a branch, "
        "realistic style, game-ready 3D model."
    )

    payload = {
        "Prompt": prompt
        # Aquí puedes añadir parámetros opcionales si quieres refinar.
    }

    resp = call_hunyuan_api("SubmitHunyuanTo3DProJob", payload)
    job_id = resp.get("JobId")
    if not job_id:
        raise RuntimeError(f"No se recibió JobId: {resp}")
    print(f"Job enviado. JobId: {job_id}")
    return job_id


def wait_and_fetch_result(job_id: str, interval: int = 5, max_tries: int = 120):
    """Hace polling de QueryHunyuanTo3DProJob hasta DONE o FAIL y muestra URLs."""
    for attempt in range(max_tries):
        resp = call_hunyuan_api("QueryHunyuanTo3DProJob", {"JobId": job_id})
        status = resp.get("Status")
        print(f"Intento {attempt+1}: estado = {status}")

        if status == "DONE":
            print("✅ Modelo generado.")
            files = resp.get("ResultFile3Ds", [])
            if not files:
                print("No se devolvieron archivos en ResultFile3Ds.")
                return

            for f in files:
                f_type = f.get("Type")
                url = f.get("Url")
                preview = f.get("PreviewImageUrl")
                print(f"- Tipo: {f_type}")
                print(f"  URL: {url}")
                if preview:
                    print(f"  Preview: {preview}")

            # (Opcional) descarga automática del primer archivo de modelo:
            # descomenta si quieres guardarlo localmente.

            first = files[0]
            if first.get("Url"):
                download_model(
                    first["Url"], f"parrot_model.{first.get('Type', 'glb')}")

            return

        if status == "FAIL":
            print("❌ La tarea ha fallado.")
            print("Código:", resp.get("ErrorCode"))
            print("Mensaje:", resp.get("ErrorMessage"))
            return

        # WAIT / RUN → seguimos consultando
        time.sleep(interval)

    print("⏹ Se alcanzó el número máximo de intentos sin que termine la tarea.")


def download_model(url: str, filename: str):
    """Función sencilla para descargar el modelo a disco."""
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Modelo descargado como: {filename}")


def main():
    job_id = submit_parrot_job()
    wait_and_fetch_result(job_id)


if __name__ == "__main__":
    main()
