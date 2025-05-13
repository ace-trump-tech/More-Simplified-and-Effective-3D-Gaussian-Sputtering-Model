import cv2
import numpy as np

def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out


class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        # cp = self._img.copy()
        # w = self._width
        # self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        # self._img[:, w/8:, :] = cp[:, :7*w/8, :]

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ], np.float32)
        K_inv = np.linalg.inv(K)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp


def main():
    img_name = r'E:\data\VIGOR\NewYork\panorama\--wTtt97XIlRI6FDhnobNA,40.765495,-73.975664,.jpg'  # 替换为你的图像文件名
    FOV = 90  # 视场角，单位度
    height = 1024  # 输出图像的高度
    width = 1024  # 输出图像的宽度
    equirectangular = Equirectangular(img_name)
    # 定义不同视角的THETA和PHI值
    thetas = [i for i in range(0,361,45)]  # 从0到360，包括0和360，共361个数
    phis = [j for j in range(0, 1,1)]  # 从-90到90，包括-90和90，共181个数
    # 遍历不同的视角
    for THETA in thetas:
        for PHI in phis:
            persp_img = equirectangular.GetPerspective(FOV, THETA, PHI, height, width)
            # 构建输出文件名
            output_filename = f'D:/code/dust3r-main/output/outdoor/perspective_image_theta{THETA}_phi{PHI}.jpg'
            cv2.imwrite(output_filename, persp_img)
            print(f'Saved: {output_filename}')

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()