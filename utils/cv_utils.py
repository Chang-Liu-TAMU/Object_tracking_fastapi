from PIL import ImageColor
import cv2


def draw_tracks(detection, img, track_id, p_out=None, p_in=None, config=None, color="RED", classification=2, save_dir=None, file_name=None):
    color = ImageColor.getcolor("Yellow", "RGB")[::-1]
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # cv2.cvtColor(src=img, code=cv2.COLOR_RGB2BGR, dst=img)
    # cv2.imwrite(save_dir + "/" + "orig_" + file_name, img)

    a1, b1 = config.top_left
    w, h = config.width, config.height
    a2, b2 = a1 + w, b1 + h
    try:
        for i in range(len(detection)):
            *_, x1, y1, x2, y2 = detection[i]
            id = track_id[i]
            # if int(c) == classification:
            cv2.rectangle(img=img, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(193, 205, 205), thickness=3)
            cv2.putText(img, str(id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 69, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "Coming in: {}".format(p_in), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 245, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "Going out: {}".format(p_out), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 250), 2, cv2.LINE_AA)
        cv2.cv2.rectangle(img=img, pt1=(a1, b1), pt2=(a2, int(b2)), color=(255, 191, 0), thickness=3)
    # cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB, dst=img)
    # print(type(img))
    except:
        return
    # cv2.imwrite("demo.jpg", img)
    # cv2.imwrite("./moder_family_detect.jpg", img)