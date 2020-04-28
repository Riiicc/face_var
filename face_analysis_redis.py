# coding=gbk
import sys, os, dlib, glob, numpy, pymysql, urllib, redis, json, urllib, time, requests, flask
from flask import request
from skimage import io

# 1.�����ؼ�������
predictor_path = 'shape_predictor_68_face_landmarks.dat'
# 2.����ʶ��ģ��
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'

# 1.�������������
detector = dlib.get_frontal_face_detector()
# 2.���������ؼ�������
sp = dlib.shape_predictor(predictor_path)
# 3. ��������ʶ��ģ��
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# win = dlib.image_window()
# ��ѡ����������list
descriptors = []
server = flask.Flask(__name__)


# ִ����Ա����
class Officer(object):

    def __init__(self, id, deepId):
        self.id = id
        self.deepId = deepId


def officer_to_json(obj):
    return {"id": obj.id, "deepId": obj.deepId}


def handle(obj):
    return Officer(obj['id'], obj['deepId'])


# redis����
redis_con = redis.Redis(host="182.0.0.1", password="123456!", port=7366, db=10,
                        decode_responses=True)  # redisĬ������db0


# ���ݿ�����
def get_db():
    db = pymysql.connect(host="182.0.0.1", user="root", password="123456", db="jnzn_base", port=3306)
    return db


# redis����
# ��ȡ����ִ����Ա��������Ϣ
def get_all_officer():
    return json.loads(redis_con.get("officer_face_arr"), object_hook=handle)


# ˢ�����е�������Ϣ�Ļ���
def refresh_all_officer():
    db = get_db()
    cur = db.cursor()
    # ʹ�� execute()  ����ִ�� SQL ��ѯ
    cur.execute("select id,face_id from t_user")

    # ʹ�� fetchall() ������ȡ��ѯ���

    data = cur.fetchall()
    # ���е�������Ϣ
    officerArr = []
    for offObj in data:
        if offObj[1] is not None and len(offObj[1]) != 0:
            officerTmp = Officer(offObj[0], list(map(float, list(offObj[1].split(',')))))
            officerArr.append(officerTmp)
    # �ر����ݿ�����
    db.close()
    redis_con.set("officer_face_arr", json.dumps(officerArr, default=officer_to_json))


# ͨ��id����������Ϣ
@server.route('/qtlface/updateFaceId', methods=['post'])
def refresh_officer_byid():
    userId = request.form.get('userId')
    cur = get_db().cursor()
    # ʹ�� execute()  ����ִ�� SQL ��ѯ
    # cur.execute("select id,face_id,portrait from t_officer where id=" + offid)
    cur.execute("select id,face_id,head_portrait from t_user where id=" + userId)
    cur.close()
    # ʹ�� fetchall() ������ȡ��ѯ���
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
                    # 2.�ؼ�����
                    shape = sp(img, d)
                    # ������������ͺ͹ؼ���
                    # win.clear_overlay()
                    # win.add_overlay(d)
                    # win.add_overlay(shape)
                    # 3.��������ȡ��128D����

                    face_descriptor = facerec.compute_face_descriptor(img, shape)
                    # ת��Ϊnumpy array
                    v = numpy.array(face_descriptor)
                    points_str = ','.join(map(lambda x: str(x), v))
                    # ִ��sql���
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
        # ����ŷʽ����
        for i in officerArr:
            tmpDeep = i.deepId
            dist_ = numpy.linalg.norm(tmpDeep - d_test)
            print(facepath)
            if (result_num > dist_):
                result_num = dist_
                result_id = i.id
                
    return {"id": result_id, "val": result_num}

# ����������Ƭulr��ȡУ����officer��id�������Ա�ֵ
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
        # ����ŷʽ����
        for i in officerArr:
            tmpDeep = i.deepId
            dist_ = numpy.linalg.norm(tmpDeep - d_test)
            if (result_num > dist_):
                result_num = dist_
                result_id = i.id
    return {"id": result_id, "val": result_num}


# ����ִ�����
def main():
    refresh_all_officer()
    # refresh_officer_byid('3609002825679872')
    # print(test_faceurl('C:/Users/Administrator/Desktop/upload/test1.jpg'))
    server.run(port=6996, debug=False, host='0.0.0.0')


if __name__ == '__main__':
    main()
