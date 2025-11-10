#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para probar diferentes configuraciones de región para Hunyuan3D
"""

import sys
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ai3d.v20250513 import ai3d_client, models

# Credenciales
sid = 'IKID8LNDpaBcR6ZopcEdopqbPE70jGbmlZFj'
skey = 'PcchhgfbNJ7lDUXMxMXUDi1FF2ThokTz'

# Regiones a probar
regions_to_test = [
    "",
    "ap-guangzhou",
    "ap-beijing",
    "ap-shanghai",
    "ap-chengdu",
    "ap-chongqing",
    "ap-singapore",
    "ap-hongkong",
]

print("Probando diferentes regiones para Hunyuan3D 3.0...")
print("=" * 60)

for region in regions_to_test:
    print(
        f"\nProbando región: '{region}' {'(por defecto)' if region == '' else ''}")

    try:
        cred = credential.Credential(sid, skey)
        http_profile = HttpProfile(
            endpoint="ai3d.tencentcloudapi.com",
            reqTimeout=30
        )
        client_profile = ClientProfile(httpProfile=http_profile)
        client = ai3d_client.Ai3dClient(cred, region, client_profile)

        # Intentar una petición simple
        req = models.SubmitHunyuanTo3DProJobRequest()
        payload = {
            "Prompt": "test",
            "GenerateType": "Normal",
            "ResultFormat": "GLB",
        }
        req.from_json_string('{"Prompt": "test"}')

        # Intentar enviar
        resp = client.SubmitHunyuanTo3DProJob(req)
        print(f"  ✅ ÉXITO - Esta región funciona!")
        print(f"  Response: {resp.to_json_string()}")
        break

    except Exception as e:
        error_msg = str(e)
        if "UnsupportedRegion" in error_msg:
            print(f"  ❌ Región no soportada")
        elif "AuthFailure" in error_msg or "签名" in error_msg:
            print(f"  ⚠️  Problema de autenticación: {error_msg[:100]}")
        elif "InvalidParameter" in error_msg or "InvalidAction" in error_msg:
            print(
                f"  ⚠️  Parámetros inválidos (pero la región es válida): {error_msg[:100]}")
        else:
            print(f"  ❌ Error: {error_msg[:150]}")

print("\n" + "=" * 60)
print("Si TODAS las regiones fallan con 'UnsupportedRegion':")
print("- Tus credenciales son para la API Internacional")
print("- Necesitas crear credenciales en la consola de China:")
print("  https://console.cloud.tencent.com/cam/capi")
print("- Y activar Hunyuan3D en:")
print("  https://console.cloud.tencent.com/hunyuan")
