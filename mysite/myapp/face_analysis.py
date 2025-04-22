from insightface.app import FaceAnalysis
import torch


class FaceAnalysisSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FaceAnalysisSingleton, cls).__new__(cls, *args, **kwargs)
            cls._instance.face_analysis = FaceAnalysis(allowed_modules=['detection', 'recognition', 'landmark_2d_106'],
                                                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            cls._instance.face_analysis.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
        return cls._instance

    def get_face_analysis(self):
        return self._instance.face_analysis


# 获取单例实例
face_analysis_instance = FaceAnalysisSingleton().get_face_analysis()
