import sys
import os
from owslib.wms import WebMapService
#from owslib.wmts import WebMapTileService

sys.path.append(os.path.abspath("/home/imran/code/python/creds"))
import wms_creds


IMG_SIZE = 1024
layers = ['HR.ORTHOIMAGERY.ORTHOPHOTOS', 'GEOGRAPHICALGRIDSYSTEMS.ETATMAJOR40', 'GEOGRAPHICALGRIDSYSTEMS.CASSINI',
 'GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2', 'GEOGRAPHICALGRIDSYSTEMS.MAPS.SCAN50.1950','GEOGRAPHICALGRIDSYSTEMS.MAPS.SCAN50.1950', 'ORTHOIMAGERY.ORTHOPHOTOS.1950-1965']

def build_envelopes(x0, y0, width, n):
    envs = []
    for i in range(n):
        xll = x0 + i * width 
        for j in range(n):
            yll = y0 + j * width
            xur = xll + width
            yur = yll + width
            env = (xll, yll, xur, yur)
            envs.append(env)
    return envs


def build_ortho(image_path, wms, env, size=512, layer='HR.ORTHOIMAGERY.ORTHOPHOTOS'):
    LAYER = layer
    SRS = 'EPSG:2154'
    IMG_SIZE = (size, size)
    img = wms.getmap( layers=[LAYER], srs=SRS, bbox=env, size=IMG_SIZE, format='image/jpeg')
    out = open(image_path, 'wb')
    out.write(img.read())
    out.close()

### WMS SERVER CREDS #############################################################
WMS_SERVER = wms_creds.WMS_SERVER
WMS_SERVER_GEOP = 'https://wxs.ign.fr/choisirgeoportail/geoportail/r/wms'
#WMTS_SERVER_GEOP = "http://wxs.ign.fr/choisirgeoportail/geoportail/wmts"
#################################################################################

wms = WebMapService(WMS_SERVER, version='1.3.0')
wms_geop = WebMapService(WMS_SERVER_GEOP, version='1.3.0')
#wmts_geop = WebMapTileService(WMTS_SERVER_GEOP)

#env = (x - DELTA, y - DELTA, x + DELTA, y + DELTA)
#env = (168873.9, 6846107.2, 169073.9, 6846307.2)

x, y = 675775.9,6857415.4
DELTA = 3000
envs = build_envelopes(x, y, DELTA, 5)

i = 1
# for env in envs:
#     print(f'{i}/{len(envs)}')
#     img = f'./res/img_{i}'
#     build_ortho(f'{img}_1.jpg', wms, env, size=IMG_SIZE, layer=layers[4])
#     build_ortho(f'{img}_2.jpg', wms, env, size=IMG_SIZE, layer=layers[1])
#     build_ortho(f'{img}_3.jpg', wms, env, size=IMG_SIZE, layer=layers[2])
#     build_ortho(f'{img}_4.jpg', wms_geop, env, size=IMG_SIZE, layer=layers[3])
#     i += 1
print(WMS_SERVER)
# def build_tiles(images_dir, wms, layers, x, y, n, delta=3000, img_size=512):
#     envs = build_envelopes(x, y, delta, n)
#     for i in range(len(envs)):
#         for j in range(len(layers)):
#             img = f'img_{i}'
#             output_tile = f'{images_dir}/{img}_{j}.jpg'
#             build_ortho(output_tile, wms, env, size=img_size, layer=layers[j])