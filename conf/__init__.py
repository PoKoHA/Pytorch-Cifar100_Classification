import conf.global_settings as settings

class Settings:
    def __init__(self, settings):
        for attr in dir(settings): # 객체안에 있는 메소드 나열
            if attr.isupper(): # 그중 대문자만
                setattr(self, attr, getattr(settings, attr))
                '''여기 부분 잘이해 X'''

settings = Settings(settings)
'''settings. 할때 여기서 해보는것랑 class이전에 하는것랑 차이점?'''