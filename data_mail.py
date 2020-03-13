# -*- coding: utf8  -*-
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 解决中文乱码问题
#sans-serif就是无衬线字体，是一种通用字体族。
#常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica, 中文的幼圆、隶书等等。
mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号

sender = 'bot3.0@qq.com'
passwd = "mupmgkzkdwqheeej"

class DataMail:
    def __init__(self,sender,pwd):
        self.sender = sender
        self.pwd = pwd
        self.img_text = ''
        self.img_list = []
    
    def __login(self):
        self.server = smtplib.SMTP_SSL("smtp.qq.com",port=465)
        self.server.login(self.sender, self.pwd)
        print('登录成功')
    
    def add_receiver(self, mail_list):
        self.mail_list = mail_list

    def add_tile(self, subject):
        self.subject = subject

    def add_img(self, path_list):
        self.img_list = path_list
        for cnt, img in enumerate(path_list, start=1):
            self.img_text += "<img src=\"cid:image%d\" alt=\"image%d\" align=\"center\" width=100%% >"%(cnt, cnt)

    
    def send(self):
        text = '<html><body>%s</body></html>'%self.img_text
        msg = MIMEMultipart()
        # 添加邮件内容
        content = MIMEText(text, _subtype='html', _charset='utf8')
        msg.attach(content)

        msg['Subject'] = self.subject
        msg['To'] = ';'.join(self.mail_list)
        msg['From'] = self.sender

        # 添加图片
        for cnt, path in enumerate(self.img_list, start=1):
            with open(path, 'rb') as fd:
                img = MIMEImage(fd.read())
            img.add_header('Content-ID', 'image%d'%cnt)
            msg.attach(img)

        self.__login()
        self.server.sendmail(self.sender, self.mail_list, msg.as_string())
        print('发送成功')






if __name__ == '__main__':
    path_list = []
    for i in range(1,10):
        name = 'image%d.png'%i
        path_list.append(name)
        x = np.random.randint(1,10,100)
        y = np.random.randint(1,10,100)
        plt.plot(x, y, label='rand')
        plt.xlabel('x label')
        plt.ylabel('y label')
        plt.title(name)
        plt.savefig(name)
    mail_list = ['zhangwei3.0@qq.com']
    m = DataMail(sender, passwd)
    m.add_receiver(mail_list)
    m.add_tile('图片测试')
    m.add_img(path_list)
    m.send()