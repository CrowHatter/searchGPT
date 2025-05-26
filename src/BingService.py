import os
import re
import concurrent.futures
import pandas as pd
import requests
import yaml

from Util import setup_logger, get_project_root, storage_cached
from text_extract.html.beautiful_soup import BeautifulSoupSvc
from text_extract.html.trafilatura import TrafilaturaSvc

logger = setup_logger('BingService')


class BingService:
    def __init__(self, config):
        self.config = config
        extract_svc = self.config.get('source_service').get('bing_search').get('text_extract')
        if extract_svc == 'trafilatura':
            self.txt_extract_svc = TrafilaturaSvc()
        elif extract_svc == 'beautifulsoup':
            self.txt_extract_svc = BeautifulSoupSvc()

    @storage_cached('bing_search_website', 'search_text')
    def call_bing_search_api(self, search_text: str) -> pd.DataFrame:
        logger.info("BingService.call_bing_search_api. query: " + search_text)
        subscription_key = self.config.get('source_service').get('bing_search').get('subscription_key')
        endpoint = self.config.get('source_service').get('bing_search').get('end_point') + "/v7.0/search"
        mkt = self.config.get('general').get('language')
        params = {'q': search_text, 'mkt': mkt}
        headers = {'Ocp-Apim-Subscription-Key': subscription_key}

        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()

            columns = ['name', 'url', 'snippet']
            if response.json().get('webPages'):
                website_df = pd.DataFrame(response.json()['webPages']['value'])[columns]
                website_df['url_id'] = website_df.index + 1
                website_df = website_df[:self.config.get('source_service').get('bing_search').get('result_count')]
            else:
                website_df = pd.DataFrame(columns=columns + ['url_id'])
        except Exception as ex:
            raise ex
        return website_df

    def call_urls_and_extract_sentences(self, website_df) -> pd.DataFrame:
        """
        :param:
            website_df: one row = one website with url
                name: website title name
                url: url
                snippet: snippet of the website given by BingAPI
        :return:
            text_df: one row = one website sentence
            columns:
                name: website title name
                url: url
                snippet: snippet of the website given by BingAPI
                text: setences extracted from the website
        """
        logger.info(f"BingService.call_urls_and_extract_sentences. website_df.shape: {website_df.shape}")
        name_list, url_list, url_id_list, snippet_list, text_list = [], [], [], [], []
        for index, row in website_df.iterrows():
            logger.info(f"Processing url: {row['url']}")
            sentences = self.extract_sentences_from_url(row['url'])
            for text in sentences:
                word_count = len(re.findall(r'\w+', text))  # approximate number of words
                if word_count < 8:
                    continue
                name_list.append(row['name'])
                url_list.append(row['url'])
                url_id_list.append(row['url_id'])
                snippet_list.append(row['snippet'])
                text_list.append(text)
        text_df = pd.DataFrame(data=zip(name_list, url_list, url_id_list, snippet_list, text_list),
                               columns=['name', 'url', 'url_id', 'snippet', 'text'])
        return text_df

    def call_one_url(self, website_tuple):
        name, url, snippet, url_id = website_tuple
        logger.info(f"Processing url: {url}")
        sentences = self.extract_sentences_from_url(url)
        logger.info(f"  receive sentences: {len(sentences)}")
        return sentences, name, url, url_id, snippet

    @storage_cached('bing_search_website_content', 'website_df')
    def call_urls_and_extract_sentences_concurrent(self, website_df):
        logger.info(f"BingService.call_urls_and_extract_sentences_async. website_df.shape: {website_df.shape}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(self.call_one_url, website_df.itertuples(index=False)))
        name_list, url_list, url_id_list, snippet_list, text_list = [], [], [], [], []
        for result in results:
            sentences, name, url, url_id, snippet = result
            sentences = sentences[:self.config['source_service']['bing_search']['sentence_count_per_site']]  # filter top N only for stability
            for text in sentences:
                word_count = len(re.findall(r'\w+', text))  # approximate number of words
                if word_count < 8:
                    continue
                name_list.append(name)
                url_list.append(url)
                url_id_list.append(url_id)
                snippet_list.append(snippet)
                text_list.append(text)
        text_df = pd.DataFrame(data=zip(name_list, url_list, url_id_list, snippet_list, text_list),
                               columns=['name', 'url', 'url_id', 'snippet', 'text'])
        return text_df

    def extract_sentences_from_url(self, url):
        # Fetch the HTML content of the page
        try:
            response = requests.get(url, timeout=3)
        except:
            logger.error(f"Failed to fetch url: {url}")
            return []
        html_content = response.text

        # Use BeautifulSoup to parse the HTML and extract the text
        extract_text = self.txt_extract_svc.extract_from_html(html_content)
        return extract_text


import re
import openai
# ... ❶ 此處保留檔案原有 import 與其他程式碼 ...

# ====== 新增：基本同義詞正規化 ======
def _normalize_user_query(q: str) -> str:
    """
    將常見口語描述先替換成正規專有名詞，
    以提高 GPT 與 Bing 的命中率。
    """
    synonym_map = {
        r"\blight\s*blade\b": "lightsaber",
        r"\bkind of force\b": "the Force",
        r"\bforce power\b": "the Force",
        r"\bmagic stick\b": "wand",
        r"\bresume\b": "CV curriculum vitae",
        # ↑ 可視情況自行擴充
    }
    for pat, repl in synonym_map.items():
        q = re.sub(pat, repl, q, flags=re.I)
    return q.strip()


# ====== 取代舊版 rewrite_query_with_gpt ======
def rewrite_query_with_gpt(original_query: str, config: dict) -> str:
    """
    產出「短、命中率高」的英文搜尋關鍵字。
    1) 先做同義詞正規化
    2) 請 GPT：若能辨識實體 ➜ 只回 'Star Wars lightsaber'
                否則 ➜ 回 ≤8 字的關鍵字
    """
    try:
        openai_cfg   = config.get("llm_service", {}).get("openai_api", {})
        openai.api_key = openai_cfg.get("api_key")
        model        = openai_cfg.get("model", "gpt-3.5-turbo")
        # 固定輸出風格
        temperature  = 0.2
        max_tokens   = 16   # 最多約 10 個英文單字

        # 1️⃣ 預先正規化
        normalized_q = _normalize_user_query(original_query)

        system_msg = (
            "You are an expert search assistant. "
            "Convert the user's request into a VERY SHORT (<=10 words) English web-search query.\n"
            "• If the request clearly refers to a well-known entity (movie, book, brand, person, event, concept), "
            "  output ONLY that entity name.\n"
            "• Otherwise, output concise English keywords (<=8 words).\n"
            "Output MUST be a single line, no quotes, no commentary."
        )

        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": normalized_q}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0
        )

        return resp.choices[0].message.content.strip()

    except Exception as e:
        # 若 GPT 失敗，就回原始查詢
        logger.warning(f"GPT query rewriting failed, fallback to original input. Error: {e}")
        return original_query


if __name__ == '__main__':
    import os
    import yaml
    from Util import get_project_root
    from BingService import BingService

    # Load config
    with open(os.path.join(get_project_root(), 'src/config/config.yaml'), encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    service = BingService(config)

    original_query = 'What is that movie where someone fixes solar panels in space?'
    rewritten_query = rewrite_query_with_gpt(original_query, config)
    print(f"User Input: {original_query}")
    print(f"GPT-Rewritten Query: {rewritten_query}")

    website_df = service.call_bing_search_api(rewritten_query)
    print("===========Website df:============")
    print(website_df)

    text_df = service.call_urls_and_extract_sentences_concurrent(website_df)
    print("===========text df:============")
    print(text_df)
