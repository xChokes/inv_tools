#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import base64
import argparse
from pathlib import Path
import requests
from PIL import Image
import io

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ai3d.v20250513 import ai3d_client, models


def optimize_image(p: Path, max_size_kb=500) -> bytes:
    """Optimiza la imagen reduciendo tamaño si es necesario"""
    if not p.is_file():
        print(f"No encuentro la imagen: {p}", file=sys.stderr)
        sys.exit(1)

    img = Image.open(p)

    # Convertir a RGB si es necesario
    if img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()
                         [-1] if img.mode in ('RGBA', 'LA') else None)
        img = background

    # Reducir tamaño si la imagen es muy grande
    max_dimension = 1024
    if max(img.size) > max_dimension:
        ratio = max_dimension / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        print(f"Imagen redimensionada a {new_size}")

    # Comprimir y ajustar calidad
    output = io.BytesIO()
    quality = 85
    img.save(output, format='JPEG', quality=quality, optimize=True)

    # Si aún es muy grande, reducir más la calidad
    while output.tell() > max_size_kb * 1024 and quality > 50:
        output = io.BytesIO()
        quality -= 10
        img.save(output, format='JPEG', quality=quality, optimize=True)

    size_kb = output.tell() / 1024
    print(f"Tamaño de imagen: {size_kb:.1f} KB (calidad: {quality})")

    return output.getvalue()


def to_b64(p: Path) -> str:
    """Convierte imagen optimizada a base64"""
    img_bytes = optimize_image(p)
    return base64.b64encode(img_bytes).decode("utf-8")


def download(url: str, out_path: Path):
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def main():
    ap = argparse.ArgumentParser(
        description="Hunyuan3D-3 Pro: frontal(+trasera opcional) -> model.glb")
    ap.add_argument("--front", required=True, help="Imagen frontal local")
    ap.add_argument("--back", help="Imagen trasera local (opcional)")
    ap.add_argument("--out-dir", default="hunyuan3d3_output")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Credenciales desde entorno
    sid = 'IKID8LNDpaBcR6ZopcEdopqbPE70jGbmlZFj'
    # sid = 'IKIDMaeXFoCNK4by9iBUXTEqGVhlfOtCKFxW'
    skey = 'PcchhgfbNJ7lDUXMxMXUDi1FF2ThokTz'
    # skey = '1385413541'
    # ap-singapore es la única región que acepta la API
    region = "ap-singapore"

    if not sid or not skey:
        print("Faltan TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY.",
              file=sys.stderr)
        sys.exit(1)

    # Cliente oficial: él se encarga de la firma TC3
    cred = credential.Credential(sid, skey)
    http_profile = HttpProfile(
        endpoint="ai3d.tencentcloudapi.com",
        reqTimeout=300  # Timeout de 5 minutos (300 segundos)
    )
    client_profile = ClientProfile(httpProfile=http_profile)
    client = ai3d_client.Ai3dClient(cred, region, client_profile)

    # 1) Submit
    print("Procesando imagen frontal...")
    front_b64 = to_b64(Path(args.front))

    req = models.SubmitHunyuanTo3DProJobRequest()
    payload = {
        "ImageBase64": front_b64,
        "GenerateType": "Normal",
        "EnablePBR": False,
        "FaceCount": 200000,
        "ResultFormat": "GLB",
    }

    # Agregar imagen trasera si se proporciona
    if args.back:
        print("Procesando imagen trasera...")
        back_b64 = to_b64(Path(args.back))
        payload["MultiViewImages"] = [
            {
                "ViewType": "back",
                "ViewImageBase64": back_b64,
            }
        ]

    req.from_json_string(json.dumps(payload))

    print("Enviando solicitud a Tencent Cloud (esto puede tardar)...")
    try:
        resp = client.SubmitHunyuanTo3DProJob(req)
    except Exception as e:
        print(f"\n❌ Error al enviar la solicitud: {e}", file=sys.stderr)
        print("\nPosibles soluciones:")
        print("1. Verifica tu conexión a internet")
        print("2. Asegúrate de que tus credenciales son válidas para la región de China")
        print("3. Intenta usar una VPN si tienes problemas de conectividad con China")
        print("4. Verifica que el servicio Hunyuan3D esté activado en tu cuenta")
        sys.exit(1)
    data = json.loads(resp.to_json_string())
    (out_dir / "submit.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    job_id = data.get("JobId")
    if not job_id:
        print("No hay JobId en la respuesta. Revisa submit.json", file=sys.stderr)
        sys.exit(1)

    print("JobId:", job_id)

    # 2) Poll
    qreq = models.QueryHunyuanTo3DProJobRequest()
    qreq.JobId = job_id

    deadline = time.time() + 30 * 60  # 30 min máx
    while True:
        qresp = client.QueryHunyuanTo3DProJob(qreq)
        qdata = json.loads(qresp.to_json_string())
        (out_dir / "last_query.json").write_text(json.dumps(qdata, indent=2), encoding="utf-8")
        status = qdata.get("Status")
        print("Estado:", status)
        if status in ("DONE", "FAIL") or time.time() > deadline:
            break
        time.sleep(10)

    if status != "DONE":
        print("La tarea no terminó bien. Mira last_query.json", file=sys.stderr)
        sys.exit(2)

    # 3) Coger primer GLB
    files = qdata.get("ResultFile3Ds") or []
    glb_url = None
    for f in files:
        if f.get("Type", "").upper() == "GLB" and f.get("Url"):
            glb_url = f["Url"]
            break
    if not glb_url and files:
        glb_url = files[0].get("Url")

    if not glb_url:
        print("No encontré URL de modelo en ResultFile3Ds.", file=sys.stderr)
        sys.exit(3)

    out_path = out_dir / "model.glb"
    download(glb_url, out_path)
    print("Modelo 3D guardado en:", out_path)


if __name__ == "__main__":
    main()
