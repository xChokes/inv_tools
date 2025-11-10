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
        description="Tencent Hunyuan 3D Global Pro: prompt -> modelo 3D (mínimo)."
    )
    parser.add_argument("--prompt", required=True,
                        help="Texto que describe el modelo 3D.")
    parser.add_argument("--out-dir", default="hy3d_prompt_min_output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Credenciales (clave estándar de Tencent Cloud)
    # Credenciales desde entorno
    sid = 'IKID8LNDpaBcR6ZopcEdopqbPE70jGbmlZFj'
    # sid = 'IKIDMaeXFoCNK4by9iBUXTEqGVhlfOtCKFxW'
    skey = 'PcchhgfbNJ7lDUXMxMXUDi1FF2ThokTz'
    # skey = '1385413541'
    region = "ap-guangzhou"

    if not sid or not skey:
        print("Faltan TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY.",
              file=sys.stderr)
        sys.exit(1)

    # Cliente oficial: maneja firma TC3 + versión internamente
    cred = credential.Credential(sid, skey)
    http_profile = HttpProfile(endpoint="hunyuan.intl.tencentcloudapi.com")
    client_profile = ClientProfile(httpProfile=http_profile)
    client = ai3d_client.Ai3dClient(cred, region, client_profile)

    # 1) SubmitHunyuanTo3DProJob - SOLO Prompt
    submit_req = models.SubmitHunyuanTo3DProJobRequest()
    payload = {
        "Prompt": args.prompt
        # Nada más. Si esto falla con ServerError, es cosa del backend.
    }
    submit_req.from_json_string(json.dumps(payload, ensure_ascii=False))

    try:
        submit_resp = client.SubmitHunyuanTo3DProJob(submit_req)
    except Exception as e:
        # Log crudo de error del SDK
        err_path = out_dir / "submit_error.txt"
        err_path.write_text(str(e), encoding="utf-8")
        print("Error al hacer Submit. Detalle en:", err_path, file=sys.stderr)
        sys.exit(1)

    submit_data = json.loads(submit_resp.to_json_string())
    (out_dir / "submit_response.json").write_text(
        json.dumps(submit_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Si aquí hay Code != 0, el problema está descrito en esta propia respuesta
    if "Error" in submit_data:
        print("Submit devolvió Error. Mira submit_response.json", file=sys.stderr)
        sys.exit(1)

    job_id = submit_data.get("JobId")
    if not job_id:
        print("No hay JobId en la respuesta. Mira submit_response.json",
              file=sys.stderr)
        sys.exit(1)

    print("JobId:", job_id)

    # 2) QueryHunyuanTo3DProJob hasta DONE/FAIL
    query_req = models.QueryHunyuanTo3DProJobRequest()
    query_req.JobId = job_id

    deadline = time.time() + 30 * 60  # 30 min máximo
    while True:
        qresp = client.QueryHunyuanTo3DProJob(query_req)
        qdata = json.loads(qresp.to_json_string())
        (out_dir / "last_query.json").write_text(
            json.dumps(qdata, indent=2, ensure_ascii=False), encoding="utf-8"
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

    # 3) Descargar el primer archivo de ResultFile3Ds
    files = qdata.get("ResultFile3Ds") or []
    if not files:
        print(
            "No hay ResultFile3Ds en la respuesta. Revisa last_query.json.", file=sys.stderr)
        sys.exit(3)

    url = None
    for f in files:
        if f.get("Url"):
            url = f["Url"]
            break

    if not url:
        print("No encontré URL en ResultFile3Ds. Revisa last_query.json.",
              file=sys.stderr)
        sys.exit(4)

    out_path = out_dir / "model.glb"
    download(url, out_path)
    print("Modelo 3D guardado en:", out_path)


if __name__ == "__main__":
    main()
