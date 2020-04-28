# coding=gbk
import sys, os, dlib, glob, numpy, pymysql, urllib, redis, json, urllib, time, requests, flask
from flask import request
from skimage import io

# 1.人脸关键点检测器
predictor_path = 'shape_predictor_68_face_landmarks.dat'
# 2.人脸识别模型
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'

# 1.加载正脸检测器
detector = dlib.get_frontal_face_detector()
# 2.加载人脸关键点检测器
sp = dlib.shape_predictor(predictor_path)
# 3. 加载人脸识别模型
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# win = dlib.image_window()
# 候选人脸描述子list
descriptors = []
server = flask.Flask(__name__)


# 执法人员类型
class Officer(object):

    def __init__(self, id, deepId):
        self.id = id
        self.deepId = deepId


def officer_to_json(obj):
    return {"id": obj.id, "deepId": obj.deepId}


def handle(obj):
    return Officer(obj['id'], obj['deepId'])


# redis链接
redis_con = redis.Redis(host="182.0.0.1", password="123456!", port=7366, db=10,
                        decode_responses=True)  # redis默认连接db0


# 数据库链接
def get_db():
    db = pymysql.connect(host="182.0.0.1", user="root", password="123456", db="jnzn_base", port=3306)
    return db


# redis集合
# 获取所有执法人员的人脸信息
def get_all_officer():
    return json.loads(redis_con.get("officer_face_arr"), object_hook=handle)


# 刷新所有的人脸信息的缓存
def refresh_all_officer():
    db = get_db()
    cur = db.cursor()
    # 使用 execute()  方法执行 SQL 查询
    cur.execute("select id,face_id from t_user")

    # 使用 fetchall() 方法获取查询结果

    data = cur.fetchall()
    # 所有的人脸信息
    officerArr = []
    for offObj in data:
        if offObj[1] is not None and len(offObj[1]) != 0:
            officerTmp = Officer(offObj[0], list(map(float, list(offObj[1].split(',')))))
            officerArr.append(officerTmp)
    # 关闭数据库连接
    db.close()
    redis_con.set("officer_face_arr", json.dumps(officerArr, default=officer_to_json))


# 通过id更新人脸信息
@server.route('/qtlface/updateFaceId', methods=['post'])
def refresh_officer_byid():
    userId = request.form.get('userId')
    cur = get_db().cursor()
    # 使用 execute()  方法执行 SQL 查询
    # cur.execute("select id,face_id,portrait from t_officer where id=" + offid)
    cur.execute("select id,face_id,head_portrait from t_user where id=" + userId)
    cur.close()
    # 使用 fetchall() 方法获取查询结果
    data = cur.fetchall()
    if (len(data) != 0):
        offObj = data[0]
        print(userId)
        filename = '/qiantaolu/' + str(offObj[0]) + '.jpg'
        if offObj[2] is not None and len(offObj[2]) != 0:
            try:
                urllib.request.urlretrieve(offObj[2], filename)
                img = io.imread(filename)
                dets = detector(img, 1)
                print('=======')
                for k, d in enumerate(dets):
                    print('=======1111')
                    # 2.关键点检测
                    shape = sp(img, d)
                    # 画出人脸区域和和关键点
                    # win.clear_overlay()
                    # win.add_overlay(d)
                    # win.add_overlay(shape)
                    # 3.描述子提取，128D向量

                    face_descriptor = facerec.compute_face_descriptor(img, shape)
                    # 转换为numpy array
                    v = numpy.array(face_descriptor)
                    points_str = ','.join(map(lambda x: str(x), v))
                    # 执行sql语句
                    db = get_db()
                    cur1 = db.cursor()
                    query1 = "update t_user set face_id = '" + points_str + "' where id = '" + userId + "'"
                    print(query1)
                    cur1.execute(query1)
                    cur1.close()
                    db.commit()
                    db.close()
            except Exception as e:
                print(e)
    refresh_all_officer()
    return {"id": "12312"}

def test_faceurl(facepath):
    mst = int(round(time.time() * 1000))
    officerArr = get_all_officer()
    img = io.imread(facepath)
    dets = detector(img, 1)
    dist = []
    result_id = 0
    result_num = 1
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        d_test = numpy.array(face_descriptor)
        # 计算欧式距离
        for i in officerArr:
            tmpDeep = i.deepId
            dist_ = numpy.linalg.norm(tmpDeep - d_test)
            print(facepath)
            if (result_num > dist_):
                result_num = dist_
                result_id = i.id
                
    return {"id": result_id, "val": result_num}

# 根据人脸照片ulr获取校验后的officer的id和人脸对比值
@server.route('/qtlface/vertifyByPicurl', methods=['post'])
def get_id_by_faceurl():
    facepath = request.form.get('facepath')
    companyId = request.form.get('companyId')
    mst = int(round(time.time() * 1000))
    officerArr = get_all_officer()
    img = io.imread(facepath)
    dets = detector(img, 1)
    dist = []
    result_id = 0
    result_num = 1
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        d_test = numpy.array(face_descriptor)
        # 计算欧式距离
        for i in officerArr:
            tmpDeep = i.deepId
            dist_ = numpy.linalg.norm(tmpDeep - d_test)
            if (result_num > dist_):
                result_num = dist_
                result_id = i.id
    return {"id": result_id, "val": result_num}


# 函数执行入口
def main():
    refresh_all_officer()
    # refresh_officer_byid('3609002825679872')
    # print(test_faceurl('C:/Users/Administrator/Desktop/upload/test1.jpg'))
    server.run(port=6996, debug=False, host='0.0.0.0')


if __name__ == '__main__':
    main()
