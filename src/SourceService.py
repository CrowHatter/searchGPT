import glob
import os
import pandas as pd

from BingService import BingService, rewrite_query_with_gpt
from Util import setup_logger
from text_extract.doc import support_doc_type, doc_extract_svc_map
from text_extract.doc.abc_doc_extract import AbstractDocExtractSvc
from website.sender import Sender, MSG_TYPE_SEARCH_STEP

logger = setup_logger('SourceModule')


class SourceService:
    def __init__(self, config, sender: Sender = None):
        self.config = config
        self.sender = sender

    # --------------------------- Bing 相關 --------------------------- #
    def _call_bing_and_collect(self, bing_service: BingService, query: str):
        """
        包一層方便呼叫 Bing API，也順便把前端進度訊息丟給 websocket。
        """
        if self.sender is not None:
            self.sender.send_message(MSG_TYPE_SEARCH_STEP, f"Calling Bing Search API: {query}")
        return bing_service.call_bing_search_api(search_text=query)

    def extract_bing_text_df(self, search_text):
        """
        1. 先用原始輸入查詢 Bing  
        2. 若 `use_gpt_query` 為 True，再用 GPT 重寫後的查詢再搜一次  
        3. 合併、去重並重新編號 url_id  
        4. 下載網頁並擷取句子  
        """
        if (not self.config['source_service']['is_use_source'] or
                not self.config['source_service']['is_enable_bing_search']):
            return None

        bing_service = BingService(self.config)

        # -- 原始查詢
        website_df_list = [self._call_bing_and_collect(bing_service, search_text)]

        # -- GPT 重寫查詢
        if self.config['source_service']['bing_search'].get('use_gpt_query', False):
            rewritten_query = rewrite_query_with_gpt(search_text, self.config)
            logger.info(f"rewrite_query_with_gpt: '{search_text}' -> '{rewritten_query}'")
            if rewritten_query and rewritten_query.strip() and rewritten_query.strip() != search_text.strip():
                website_df_list.append(self._call_bing_and_collect(bing_service, rewritten_query))

        # -- 合併結果、去重、重新編號 url_id
        website_df = (pd.concat(website_df_list, ignore_index=True)
                        .drop_duplicates(subset='url')
                        .reset_index(drop=True))
        website_df['url_id'] = range(1, len(website_df) + 1)

        if self.sender is not None:
            self.sender.send_message(MSG_TYPE_SEARCH_STEP, "Extracting sentences from Bing search result ...")

        # -- 擷取句子（⚠️ 必須用 keyword 傳入 website_df）
        bing_text_df = bing_service.call_urls_and_extract_sentences_concurrent(website_df=website_df)
        return bing_text_df

    # --------------------------- 文件搜尋 --------------------------- #
    def extract_doc_text_df(self, bing_text_df):
        """
        從本地文件夾抽取文字並與 Bing 結果併入同一 DataFrame
        """
        if (not self.config['source_service']['is_use_source'] or
                not self.config['source_service']['is_enable_doc_search']):
            return pd.DataFrame([])

        if self.sender is not None:
            self.sender.send_message(MSG_TYPE_SEARCH_STEP, "Extracting sentences from document")

        files_grabbed = []
        for doc_type in support_doc_type:
            pattern = os.path.join(self.config['source_service']['doc_search_path'], f"*.{doc_type}")
            files_grabbed.extend({"file_path": p, "doc_type": doc_type} for p in glob.glob(pattern))

        logger.info(f"File list: {files_grabbed}")
        doc_sentence_list = []

        start_doc_id = 1 if bing_text_df is None or len(bing_text_df) == 0 else bing_text_df['url_id'].max() + 1
        for doc_id, file in enumerate(files_grabbed, start=start_doc_id):
            extract_svc: AbstractDocExtractSvc = doc_extract_svc_map[file['doc_type']]
            sentence_list = extract_svc.extract_from_doc(file['file_path'])

            file_name = os.path.basename(file['file_path'])
            for sentence in sentence_list:
                doc_sentence_list.append({
                    'name': file_name,
                    'url': file['file_path'],
                    'url_id': doc_id,
                    'snippet': '',
                    'text': sentence
                })

        return pd.DataFrame(doc_sentence_list)
