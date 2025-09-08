import random
import numpy as np

def Plane(pts, thresh=0.05, minPoints=100, maxIteration=1000):
    """
    Find the best equation for plane of pts.

    :param pts: 3D point cloud as a 'np.array (N,3)'
    :param thresh: Threshold distance treated as inlier
    :param maxIteration: Number of max iteration for RANSAC
    :returns:
    - 'equation': Plane parameter using Ax+By+Cy+D 'np.array (1,4)'
    - 'inliers': Points from the dataset considered inliers
    """

    n_points = pts.shape[0]
    best_eq = []
    best_inliers = []
    inliers = [] 
    equation = [] 

    for it in range(maxIteration):

        # pts에서 3개의 random point 추출
        id_samples = random.sample(range(0, n_points), 3) # RANSAC의 핵심
        pt_samples = pts[id_samples]

        # 매 iteration마다 random으로 뽑은 3개의 점으로 평면 식 구하려는 목적
        # 평면을 구성하는 2개의 vector 구하기
        # A = pt2 - pt1
        # B = pt3 - pt1

        vecA = pt_samples[1, :] - pt_samples[0, :]
        vecB = pt_samples[2, :] - pt_samples[0, :]

        # A와 B의 cross product로 normal vector C 구하기
        vecC = np.cross(vecA, vecB) # 평면의 법선 벡터

        # 평면 방정식은 vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
        # 3개 중 1개의 point 사용해서 상수 k 값 구하기
        vecC = vecC / np.linalg.norm(vecC) # 평면 반사 연산에서 단위 법선벡터 사용하기 때문
        k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
        plane_eq = [vecC[0], vecC[1], vecC[2], k] # 평면 방정식 완성

        # 올바른 후보군 선정을 위해, 평면과 점 사이의 거리 구하기
        pt_id_inliers = []  # pts 전체 point 중, 평면 내부라고 판단된 점들의 id list
        dist_pt = (
            plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
        ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2) # 점과 평면 사이의 거리

        # Threshold보다 작은 distance를 가지는 경우 평면 내부의 point로 고려
        pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]

        # 현재 best inlier의 수보다 많아지면 update
        # RANSAC의 기본 개념, 이상치를 최소화하는 best model을 정의한다
        if len(pt_id_inliers) > len(best_inliers):
            best_eq = plane_eq
            best_inliers = pt_id_inliers 
        inliers = best_inliers
        equation = best_eq

    return equation, inliers