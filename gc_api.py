import json
import time
import uuid
import requests
import typing as tp
from urllib.parse import urlencode

from ssl_context import no_ssl_verification
from lib_vars import CONTOUR_API_URL, CONTOUR_OAUTH_URL


class GigaChatAPI:
    '''
    Класс, реализующий доступ к API GigaChat'а
    '''
    def __init__(self,
                 contour: tp.Optional[str]=None,
                 token: tp.Optional[str]=None,
                 auth_data: tp.Optional[str]=None,
                 api_url: tp.Optional[str]=None,
                 oauth_url: tp.Optional[str]=None,
                 rquid: tp.Optional[str]=None):
        '''
        contour: Название контура. Принимает одно из трех значениий ['sigma', 'mlspace', 'datalab']
        token: Токен для авторизации. Передается, если уже получен
        auth_data: Авторизационные данные. Client id и Client Secret, закодированные в Base64
        api_url: URL до API GigaChat 
        oauth_url: URL до сервиса авторизации GigaChat
        rquid: Уникальный идентификатор запроса. Соответствует формату uuid4.
                Параметр для журналирования входящих вызовов и разбора инцидентов.
        '''
        self.rquid = rquid if rquid is not None else str(uuid.uuid4())
        self.api_url = self.get_url(contour, api_url, CONTOUR_API_URL)
        self.__is_ext_token = token is not None
        if not self.__is_ext_token:
            self.oauth_url = self.get_url(contour, oauth_url, CONTOUR_OAUTH_URL)
            if auth_data is None:
                raise ValueError('Необходимо передать auth_data')
            self.__auth_data = auth_data
        self.token = token if self.__is_ext_token else self.get_token(self.__auth_data)


    def get_url(self,
                contour: tp.Optional[str]=None,
                url: tp.Optional[str]=None,
                url_dict: tp.Dict[str, str]=CONTOUR_API_URL) -> str:
        '''
        Метод получения URL в зависимости от переданных параметров. 
        Если передан и url, и contour, то в приоритетет возвращает url
        Parameters:
            contour: Название контура
            contour_dict: Словарь, где ключ - название контура, значение - URL
        Return:
            URL в зависимости от контура и переданного URL
        '''
        if url is None:
            if contour is not None:
                self._check_contour(contour, url_dict)
                url = url_dict[contour]
            else:
                raise ValueError('Необходимо передать либо contour, либо url')
        return url


    def _check_contour(self, 
                       contour: str,
                       contour_dict: tp.Dict[str, str]) -> None:
        '''
        Метод проверки контура, если названия контура нет в ключах словаря, то 'выкидывает' ValueError.
        Parameters:
            contour: Название контура
            contour_dict:  Словарь, где ключ - название контура, значение - URL
        '''
        if not contour in contour_dict:
            raise ValueError(f'contour не может быть {contour}.\ncontour может быть: {list(contour_dict.keys())}')
    

    def get_token(self,
                  auth_data: str) -> str:
        '''
        Метод получения Корпоративного токена. Токен необходим для авторизации для API GigaChat.
        Токен действителен в течение 30 минут
        Parameters:
            auth_data: Авторизационные данные. Client id и Client Secret, закодированные в Base64
        Return:
            Токен доступа (длинная строка формата 'CgirZ-DdmCFyRixdl9...MPQTiP-RGvlB-Rpmikw6B')
        '''
        url = self.oauth_url
        headers = {'RqUID': self.rquid,
                   'Content-Type': 'application/x-www-form-urlencoded',
                   'Authorization': f'Bearer {auth_data}'}
        body = urlencode({'scope': 'GIGACHAT_API_CORP'})
        with no_ssl_verification():
            response = requests.post(url, data=body, headers=headers)
        response.raise_for_status()
        data_dict = response.json()
        self.token_expires_time = data_dict['expires_at']
        return data_dict['access_token']


    def update_token(self, auth_data: str) -> None:
        '''
        Метод обновления Корпоративного токена. Токен необходим для авторизации для API GigaChat.
        Токен действителен в течение 30 минут
        Parameters:
            auth_data: Авторизационные данные. Client id и Client Secret, закодированные в Base64
        '''
        if not self.__is_ext_token:
            self.token = self.get_token(auth_data)


    def is_token_expires(self) -> bool:
        '''
        Метод проверки 'свежести' токена. Сравнивает текущее время и время истечения токена
        Return:
            True если токен 'протух', иначе False
        '''
        now_unix_time = int(time.time() * 1000)
        return (not self.__is_ext_token) and (now_unix_time > self.token_expires_time)


    def get_list_models(self) -> tp.List[tp.Dict[str, str]]:
        '''
        Метод получения списка доступных моделей.
        Return:
            Список словарей с данными доступных моделей. Пример словаря:
                {'id': 'GigaChat', 'object': 'model', 'owned_by': 'salutedevices'}
        '''
        if self.is_token_expires():
            self.update_token(self.__auth_data)
        url = self.api_url + '/models'
        headers = {'Accept': 'application/json',
                   'Authorization': f'Bearer {self.token}'}
        body = ''
        with no_ssl_verification():
            response = requests.get(url, data=body, headers=headers)
        response.raise_for_status()
        data_dict = response.json()
        return data_dict['data']


    def get_answer_gigachat(self,
                            messages: tp.List[tp.Dict[str, tp.Any]],
                            model: str='GigaChat:latest',
                            function_call: tp.Optional[tp.Union[str, tp.Dict[str, tp.Any]]]=None,
                            functions: tp.Optional[tp.List[tp.Dict[str, tp.Any]]]=None,
                            temperature: tp.Optional[float]=None,
                            top_p: tp.Optional[float]=None,
                            n: tp.Optional[int]=None,
                            stream: tp.Optional[bool]=None,
                            max_tokens: tp.Optional[int]=None,
                            repetition_penalty: tp.Optional[float]=None,
                            update_interval: tp.Optional[int]=None,
                            profanity_check: tp.Optional[bool]=None) -> tp.List[tp.Dict[str, tp.Any]]:
        '''
        Метод получения ответа модели с учетом переданных сообщений.
        Подробное описания каждого параметра см. в документации к API:
        https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/post-chat
        Parameters:
            messages: Список сообщений(словарей), которыми пользователь обменивался с моделью.
                Сообщения имеют следующую структуру:
                    {'role': 'user',
                     'data_for_context': [{}]
                     'content': 'Тестовое сообщение. Напиши 'капибара', если меня понимаешь.'}
            model: Название модели
            function_call: Поле, которое отвечает за то, как GigaChat будет работать с функциями.
            functions: Массив с описанием пользовательских функций
            temperature: Температура выборки.
                Значение температуры должно быть не меньше ноля. Чем выше значение, тем более случайным будет ответ модели.
                Значение по умолчанию зависит от выбранной модели и может изменяться с обновлениями модели.
            top_p: Возможные значения: >= 0 и  <= 1.
                Задает вероятностную массу токенов, которые должна учитывать модель.
                Значение по умолчанию зависит от выбранной модели и может изменяться с обновлениями модели.
            n: Количество вариантов ответов, которые нужно сгенерировать для каждого входного сообщения.
            stream: Указывает, что сообщения надо передавать по частям в потоке.
            max_tokens: Максимальное количество токенов, которые будут использованы для создания ответов.
            repetition_penalty: Количество повторений слов. Значение 1 — нейтральное значение.
                При значении больше 1 модель будет стараться не повторять слова.
                Значение по умолчанию зависит от выбранной модели и может изменяться с обновлениями модели.
            update_interval: Задает минимальный интервал в секундах, который проходит между отправкой токенов.
            profanity_check: Переключатель цензора. Если False, то цензор выключен
        Return:
            Список ответов модели. Ответ имеет следующую структуру:
                {'message': {'role': 'assistant',
                             'content': 'Здравствуйте! К сожалению, я не могу дать точный ответ на этот вопрос',
                             'data_for_context': [{}]},
                'index': 0,
                'finish_reason': 'stop'}
        '''
        if self.is_token_expires():
            self.update_token(self.__auth_data)
        url = self.api_url + '/chat/completions'
        headers = {'Content-Type': 'application/json',
                   'Accept': 'application/json',
                   'Authorization': f'Bearer {self.token}'}
        body = {'messages': messages,
                'model': model}
        if function_call is not None:
            body['function_call'] = function_call
        if functions is not None:
            body['functions'] = functions
        if n is not None:
            body['n'] = n
        if stream is not None:
            body['stream'] = stream
        if max_tokens is not None:
            body['max_tokens'] = max_tokens
        if temperature is not None:
            body['temperature'] = temperature
        if top_p is not None:
            body['top_p'] = top_p
        if repetition_penalty is not None:
            body['repetition_penalty'] = repetition_penalty
        if update_interval is not None:
            body['update_interval'] = update_interval
        if profanity_check is not None:
            body['profanity_check'] = profanity_check
        with no_ssl_verification():
            response = requests.post(url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        data_dict = response.json()
        return data_dict['choices']


    def get_embedding(self,
                      corpus: tp.List[str],
                      model: str='Embeddings',
                      profanity_check: bool=False) -> tp.List[tp.Dict[str, tp.Any]]:
        '''
        Метод получения векторных представлений соответствующих текстовых запросов.
        Подробное описания каждого параметра см. в документации к API:
        https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/post-chat
        Parameters:
            corpus: Массив строк, которые будут использованы для генерации эмбеддинга.
            model: Название модели, которая будет использована для создания эмбеддинга.
        Return:
            Возвращает список словарей с эмбедингами. Ответ имеет следующую структуру:
                {'object': 'embedding',
                'embedding': [0],
                'index': 0,
                'usage': {'prompt_tokens': 6}}
        '''
        if self.is_token_expires():
            self.update_token(self.__auth_data)
        url = self.api_url + '/embeddings'
        headers = {'Content-Type': 'application/json',
                   'Accept': 'application/json',
                   'Authorization': f'Bearer {self.token}'}
        body = {'model': model,
                'input': corpus,
                'profanity_check': profanity_check}
        with no_ssl_verification():
            response = requests.post(url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        data_dict = response.json()
        return data_dict['data']


    def get_token_count(self,
                        corpus: tp.List[str],
                        model: tp.Optional[str]='GigaChat',
                        profanity_check: bool=False) -> tp.List[tp.Dict[str, tp.Any]]:
        '''
        Метод получения количества токенов, подсчитанных заданной моделью в строках.
        Parameters:
            corpus: Массив строк, в которых надо подсчитать количество токенов.
            model: Название модели, которая будет использована для подсчета количества токенов.
        Return:
            Возвращает список словарей с ответами. Ответ имеет следующую структуру:
                {'object': 'tokens',
                'tokens': 7,
                'characters': 36}
        '''
        if self.is_token_expires():
            self.update_token(self.__auth_data)
        url = self.api_url + '/tokens/count'
        headers = {'Content-Type': 'application/json',
                   'Accept': 'application/json',
                   'Authorization': f'Bearer {self.token}'}
        body = {'model': model,
                'input': corpus,
                'profanity_check': profanity_check}
        with no_ssl_verification():
            response = requests.post(url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        data_dict = response.json()
        return data_dict
