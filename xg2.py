#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import argparse
from pathlib import Path

import requests
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ai3d.v20250513 import ai3d_client, models


def download(url: str, out_path: Path):
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def main():
    parser = argparse.ArgumentParser(
        description="Hunyuan3D-3 Pro: solo prompt de texto -> modelo 3D (GLB)"
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Descripción del objeto 3D (cualquier idioma; máximo según doc).",
    )
    parser.add_argument(
        "--out-dir",
        default="hunyuan3d3_prompt_output",
        help="Carpeta de salida.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Credenciales desde entorno
    sid = 'IKID8LNDpaBcR6ZopcEdopqbPE70jGbmlZFj'
    # sid = 'IKIDMaeXFoCNK4by9iBUXTEqGVhlfOtCKFxW'
    skey = 'PcchhgfbNJ7lDUXMxMXUDi1FF2ThokTz'
    # skey = '1385413541'
    region = "ap-singapore"

    if not sid or not skey:
        print("Faltan TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY.",
              file=sys.stderr)
        sys.exit(1)

    # Cliente oficial (se encarga de la firma y de Version=2025-05-13)
    cred = credential.Credential(sid, skey)
    http_profile = HttpProfile(endpoint="ai3d.tencentcloudapi.com")
    client_profile = ClientProfile(httpProfile=http_profile)
    client = ai3d_client.Ai3dClient(cred, region, client_profile)

    # ---------- 1) Submit: SOLO PROMPT ----------
    # OJO: según doc, para prompt-only no envíes ImageBase64/ImageUrl.
    submit_req = models.SubmitHunyuanTo3DProJobRequest()

    payload = {
        "Prompt": args.prompt,
        "GenerateType": "Normal",   # modo estándar
        "EnablePBR": True,          # PBR activado; puedes poner False si quieres más ligero
        "FaceCount": 200000,        # resolución media para ir seguros
        "ResultFormat": "GLB",      # queremos GLB directo
    }

    submit_req.from_json_string(json.dumps(payload, ensure_ascii=False))
    submit_resp = client.SubmitHunyuanTo3DProJob(submit_req)
    submit_data = json.loads(submit_resp.to_json_string())
    (out_dir / "submit_response.json").write_text(
        json.dumps(submit_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    job_id = submit_data.get("JobId")
    if not job_id:
        print("No hay JobId en la respuesta. Mira submit_response.json",
              file=sys.stderr)
        sys.exit(1)

    print("JobId:", job_id)

    # ---------- 2) Poll hasta DONE ----------
    query_req = models.QueryHunyuanTo3DProJobRequest()
    query_req.JobId = job_id

    deadline = time.time() + 30 * 60  # hasta 30 min
    while True:
        qresp = client.QueryHunyuanTo3DProJob(query_req)
        qdata = json.loads(qresp.to_json_string())
        (out_dir / "last_query.json").write_text(
            json.dumps(qdata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        status = qdata.get("Status")
        print("Estado:", status)

        if status in ("DONE", "FAIL") or time.time() > deadline:
            break

        time.sleep(10)

    if status != "DONE":
        print("La tarea no terminó correctamente. Revisa last_query.json.",
              file=sys.stderr)
        sys.exit(2)

    # ---------- 3) Descargar modelo GLB ----------
    files = qdata.get("ResultFile3Ds") or []
    if not files:
        print(
            "No hay ResultFile3Ds en la respuesta. Revisa last_query.json.", file=sys.stderr)
        sys.exit(3)

    glb_url = None
    for f in files:
        if f.get("Type", "").upper() == "GLB" and f.get("Url"):
            glb_url = f["Url"]
            break

    if not glb_url:
        # fallback al primer archivo con URL
        for f in files:
            if f.get("Url"):
                glb_url = f["Url"]
                break

    if not glb_url:
        print("No encontré URL de modelo en ResultFile3Ds.", file=sys.stderr)
        sys.exit(4)

    out_path = out_dir / "model.glb"
    download(glb_url, out_path)
    print("Modelo 3D guardado en:", out_path)


if __name__ == "__main__":
    main()
