import sys
import os
from owslib.wms import WebMapService
from owslib.wmts import WebMapTileService
from pyproj import Transformer

sys.path.append(os.path.abspath("../creds"))
import wms_creds


IMG_SIZE = 512
LAYERS = ['HR.ORTHOIMAGERY.ORTHOPHOTOS', 'GEOGRAPHICALGRIDSYSTEMS.ETATMAJOR40', 'GEOGRAPHICALGRIDSYSTEMS.CASSINI',
 'GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2', 'GEOGRAPHICALGRIDSYSTEMS.MAPS.SCAN50.1950', 'ORTHOIMAGERY.ORTHOPHOTOS.1950-1965',
 'GEOGRAPHICALGRIDSYSTEMS.MAPS.SCAN-EXPRESS.STANDARD']

# taken from https://geoservices.ign.fr/documentation/geoservices/wmts.html
# m/px for each zoom level
ZOOM_RES_L93 = {
    0: 156543.0339280410, 1: 78271.5169640205, 2: 39135.7584820102, 
    3: 19567.8792410051, 4: 9783.9396205026, 5: 4891.9698102513, 	
    6: 2445.9849051256, 7: 1222.9924525628, 8: 611.4962262814, 
    9: 305.7481131407, 10: 152.8740565704, 11: 76.4370282852, 	
    12: 38.2185141426, 13: 19.1092570713, 14: 9.5546285356, 	
    15: 4.7773142678, 16: 2.3886571339, 17: 1.1943285670, 
    18: 0.5971642835, 19: 0.2985821417, 20: 0.1492910709, 21: 0.0746455354 
}
### WMS SERVER CREDS #############################################################
WMS_SERVER = wms_creds.WMS_SERVER
WMS_SERVER_GEOP = 'https://wxs.ign.fr/choisirgeoportail/geoportail/r/wms'
WMTS_SERVER_GEOP = "http://wxs.ign.fr/choisirgeoportail/geoportail/wmts"
#################################################################################

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


def build_wms_tile(image_path, wms, env, layer, size=512):
    LAYER = layer
    SRS = 'EPSG:2154'
    IMG_DIMS = (size, size)
    img = wms.getmap( layers=[LAYER], srs=SRS, bbox=env, size=IMG_DIMS, format='image/jpeg')
    out = open(image_path, 'wb')
    out.write(img.read())
    out.close()

def build_wmst_tile(image_path, wmts, layer, x93, y93, zoom_level):
    taille_tuile = ZOOM_RES_L93[zoom_level] * 256
    X0, Y0 = wmts.tilematrixsets['PM'].tilematrix['13'].topleftcorner
    proj = Transformer.from_crs(2154, 3857, always_xy=True)
    x_m, y_m = proj.transform(x, y)
    x_g = x_m - X0
    y_g = Y0 - y_m 
    tilecol = int(x_g / taille_tuile)
    tilerow = int(y_g / taille_tuile)
    tile = wmts_geop.gettile(layer=layer, tilematrixset='PM', tilematrix=f'{zoom_level}', 
                            row=tilerow, column=tilecol, format="image/jpeg")
    out = open(image_path, 'wb')
    bytes_written = out.write(tile.read())
    out.close()


wms = WebMapService(WMS_SERVER, version='1.3.0')
wms_geop = WebMapService(WMS_SERVER_GEOP, version='1.3.0')
wmts_geop = WebMapTileService(WMTS_SERVER_GEOP)

#env = (x - DELTA, y - DELTA, x + DELTA, y + DELTA)
#env = (168873.9, 6846107.2, 169073.9, 6846307.2)

#x, y = 675775.9,6857415.4

# tile = wmts_geop.gettile(layer='GEOGRAPHICALGRIDSYSTEMS.MAPS.SCAN-EXPRESS.STANDARD', tilematrixset='PM', tilematrix='18', row=90241, column=132877, format="image/jpeg")
tiles_res = './res_tiles'
# ZOOM_MAX = 18
# for ZOOM_LEVEL in range(0, ZOOM_MAX):
#     print(ZOOM_LEVEL, "/", ZOOM_MAX - 1)
#     build_wmst_tile(f"{tiles_res}/{ZOOM_LEVEL}.jpg", wmts_geop, LAYERS[6], x, y, ZOOM_LEVEL)


#X, Y = 675775.9,6857415.4
X, Y =  498465.5,6601459.5
DELTA = 5000
envs = build_envelopes(X, Y, DELTA, 5)
# print(envs)
wms_res = './wms_examples'
for i, env in enumerate(envs, start=1):
    print(f'{i}/{len(envs)}')
    img = f'{wms_res}/{i}'
    build_wms_tile(f'{img}_1.jpg', wms, env, layer=LAYERS[4], size=IMG_SIZE)
    build_wms_tile(f'{img}_2.jpg', wms, env, layer=LAYERS[1], size=IMG_SIZE)
    build_wms_tile(f'{img}_3.jpg', wms, env, layer=LAYERS[2], size=IMG_SIZE)
    build_wms_tile(f'{img}_4.jpg', wms_geop, env, layer=LAYERS[3], size=IMG_SIZE)