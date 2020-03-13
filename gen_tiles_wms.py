import sys
import os
from owslib.wms import WebMapService
from owslib.wmts import WebMapTileService
from pyproj import Transformer


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
WMTS_SERVER_GEOP = "http://wxs.ign.fr/choisirgeoportail/geoportail/wmts"
#################################################################################

wms = WebMapService(WMS_SERVER, version='1.3.0')
wms_geop = WebMapService(WMS_SERVER_GEOP, version='1.3.0')
wmts_geop = WebMapTileService(WMTS_SERVER_GEOP)

#env = (x - DELTA, y - DELTA, x + DELTA, y + DELTA)
#env = (168873.9, 6846107.2, 169073.9, 6846307.2)

x, y = 675775.9,6857415.4
DELTA = 3000
envs = build_envelopes(x, y, DELTA, 1)


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
print(envs)
# def build_tiles(images_dir, wms, layers, x, y, n, delta=3000, img_size=512):
#     envs = build_envelopes(x, y, delta, n)
#     for i in range(len(envs)):
#         for j in range(len(layers)):
#             img = f'img_{i}'
#             output_tile = f'{images_dir}/{img}_{j}.jpg'
#             build_ortho(output_tile, wms, env, size=img_size, layer=layers[j])


zoom_res = {
0: 156543.0339280410, 1: 78271.5169640205, 2: 39135.7584820102, 
3: 19567.8792410051, 4: 9783.9396205026, 5: 4891.9698102513, 	
6: 2445.9849051256, 7: 1222.9924525628, 8: 611.4962262814, 
9: 305.7481131407, 10: 152.8740565704, 11: 76.4370282852, 	
12: 38.2185141426, 13: 19.1092570713, 14: 9.5546285356, 	
15: 4.7773142678, 16: 2.3886571339, 17: 1.1943285670, 
18: 0.5971642835, 19: 0.2985821417, 20: 0.1492910709, 21: 0.0746455354 
}

#x, y = 661729.0752875905, 6856300.088968124
ZOOM_LEVEL = 15
X0, Y0 = wmts_geop.tilematrixsets['PM'].tilematrix['13'].topleftcorner
print(int(X0), int(Y0))
proj = Transformer.from_crs(2154, 3857, always_xy=True)
x_m, y_m = proj.transform(x, y)
print(x, y, "=>", x_m, y_m)
x_g = x_m - X0
y_g = Y0 - y_m 
taille_tuile = zoom_res[ZOOM_LEVEL] * 256
tilecol = int(x_g / taille_tuile)
tilerow = int(y_g / taille_tuile)

print(tilerow, tilecol)
tile = wmts_geop.gettile(layer='GEOGRAPHICALGRIDSYSTEMS.MAPS.SCAN-EXPRESS.STANDARD', tilematrixset='PM', tilematrix=f'{ZOOM_LEVEL}', 
                          row=tilerow, column=tilecol, format="image/jpeg")
#tile = wmts_geop.gettile(layer='GEOGRAPHICALGRIDSYSTEMS.MAPS.SCAN-EXPRESS.STANDARD', tilematrixset='PM', tilematrix='18', row=90241, column=132877, format="image/jpeg")
out = open('lolknee.jpg', 'wb')
bytes_written = out.write(tile.read())
out.close()