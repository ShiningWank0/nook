"""Gemini API クライアント。"""

import os
from typing import Dict, List, Optional, Union, Any

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()


class GeminiClient:
    """
    Gemini API との通信を担当するクライアントクラス。
    
    Parameters
    ----------
    api_key : str, optional
        Gemini APIキー。指定しない場合は環境変数から取得。
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        GeminiClientを初期化します。
        
        Parameters
        ----------
        api_key : str, optional
            Gemini APIキー。指定しない場合は環境変数から取得。
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set as an environment variable")
        
        # Gemini APIの設定
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_content(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        テキストを生成します。
        
        Parameters
        ----------
        prompt : str
            生成のためのプロンプト。
        system_instruction : str, optional
            システム指示。
        temperature : float, default=0.7
            生成の多様性を制御するパラメータ。
        max_tokens : int, default=1000
            生成するトークンの最大数。
            
        Returns
        -------
        str
            生成されたテキスト。
        """
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        full_prompt = ""
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n"
        full_prompt += prompt
        
        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        return response.text
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def create_chat(
        self,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        チャットセッションを作成します。
        
        Parameters
        ----------
        system_instruction : str, optional
            システム指示。
            
        Returns
        -------
        Dict[str, Any]
            チャットセッション情報。
        """
        chat = self.model.start_chat(
            history=[]
        )
        
        if system_instruction:
            # Gemini APIではシステムメッセージを