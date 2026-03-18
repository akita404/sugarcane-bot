import os
import datetime
import numpy as np
import tensorflow as tf
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage
from PIL import Image

app = Flask(__name__)

# --- 1. ใส่ค่าจาก Line Developer ของคุณตรงนี้ ---
LINE_ACCESS_TOKEN = '6zTH+lnX/SOjERQaZQ23CxzNRdZWhxg/pM05sbS4OSlKaprPIXXgUYeSB6ZCBE/t+9fRe0jM9zRVGWN9PAiKcJ9utjSeP1THMGYLfAqqO632K9gSS/VJ+h7mFRSZ35iPAw4HW2X8LUBm4jbed2H5DgdB04t89/1O/w1cDnyilFU='
LINE_SECRET = '177abc35ad4b3fb3ea47c2f2b8bbf728'

line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_SECRET)

# --- 2. โหลดโมเดลที่เทรนเสร็จแล้ว ---
# ตรวจสอบว่าไฟล์ชื่อนี้อยู่ในโฟลเดอร์ models จริงไหม
model = tf.keras.models.load_model('models/sugarcane_model.h5')

# 1. แก้ไขรายชื่อคลาส (ต้องมี 5 ชื่อ และสะกดให้ตรงกับชื่อโฟลเดอร์ใน dataset)
class_names = ['Healthy', 'Rust', 'Smut', 'White_leaf']

class_names_th = {
    'Healthy': 'ไม่พบโรค (ปกติ)',
    'Rust': 'โรคราสนิม',
    'Smut': 'โรคแส้ดำ',
    'White_leaf': 'โรคใบขาว'
}

# ข้อมูลการรักษา
treatments = {
    'Smut': """ โรคแส้ดำ (Sugarcane Smut)
1. ลักษณะโรค:
- มียอดสีดำยาวคล้ายแส้โผล่ออกจากยอดอ้อย
- ต้นแคระแกร็น แตกกอมากผิดปกติ
2. การรักษา:
- ถอนและทำลายต้นที่เป็นโรคทันที
3. การป้องกัน:
- ใช้ท่อนพันธุ์ปลอดโรค
- ปลูกพันธุ์อ้อยที่ต้านทานโรค""",

    'Rust': """ โรคราสนิม (Sugarcane Rust)
1. ลักษณะโรค:
- มีจุดหรือแผลสีส้ม–น้ำตาลบนใบ
- ใบเหลืองหรือแห้งก่อนเวลา
2. การรักษา:
- ตัดและทำลายใบที่เป็นโรค
- หากระบาดมากสามารถใช้สารป้องกันกำจัดเชื้อรา
3. การป้องกัน:
- ปลูกพันธุ์อ้อยที่ต้านทานโรค
- ลดความชื้นและเพิ่มการระบายอากาศในแปลง""",

    'White_leaf': """ โรคใบขาว (Sugarcane White Leaf Disease)
1. ลักษณะโรค:
- ใบมีสีขาวหรือซีดผิดปกติ
- ต้นแคระแกร็น แตกกอจำนวนมาก
2. การรักษา:
- ถอนและทำลายต้นที่เป็นโรคทันที
3. การป้องกัน:
- ใช้ท่อนพันธุ์ปลอดโรค
- ควบคุมแมลงพาหะ เช่น เพลี้ยจักจั่น""",

    'Healthy': """ ไม่พบโรค (Healthy)
1. ลักษณะใบปกติ:
- ใบสีเขียวสม่ำเสมอ
- ไม่มีจุดหรือแผลผิดปกติ
2. การดูแล:
- ตรวจแปลงอ้อยเป็นประจำ
3. การป้องกัน:
- ใช้พันธุ์อ้อยแข็งแรง
- จัดการน้ำและธาตุอาหารให้เหมาะสม"""
}


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # 1. ดึงรูปภาพจาก Line
    message_content = line_bot_api.get_message_content(event.message.id)
    file_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{event.source.user_id}.jpg"
    file_path = os.path.join('uploads', file_name)
    
    with open(file_path, 'wb') as f:
        for chunk in message_content.iter_content():
            f.write(chunk)

    # 2. นำรูปมาวิเคราะห์
    img = Image.open(file_path).convert('RGB').resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    predictions = model.predict(img_array)
    score = predictions[0]
    result_idx = np.argmax(score)
    result_class = class_names[result_idx]
    result_th = class_names_th.get(result_class, result_class) 
    confidence = 100 * np.max(score)

    # 3. ส่งผลลัพธ์กลับ
    if confidence < 70: # ปรับค่าความมั่นใจที่เหมาะสม (60-70)
        response_text = f"⚠️ AI ไม่ค่อยมั่นใจในรูปนี้ (มั่นใจ {confidence:.2f}%)\nกรุณาถ่ายภาพใบอ้อยให้ชัดเจนและใกล้ขึ้นครับ"
    else:
        info = treatments.get(result_class, "ไม่มีข้อมูลการรักษา")
        response_text = f"📊 ผลการวิเคราะห์: {result_th}\n✨ ความมั่นใจ: {confidence:.2f}%\n\n💡 คำแนะนำ:\n{info}"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_text))

if __name__ == "__main__":
    app.run(port=5000)