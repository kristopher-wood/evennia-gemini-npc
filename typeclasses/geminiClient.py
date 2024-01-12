from evennia.contrib.rpg.llm.llm_client import SimpleResponseReceiver, QuietHTTP11ClientFactory, StringProducer
import json
import requests
from django.conf import settings
from evennia import logger
from evennia.utils.utils import make_iter
from twisted.internet import defer, protocol, reactor
from twisted.internet.defer import inlineCallbacks
from twisted.web.client import Agent, HTTPConnectionPool, _HTTP11ClientFactory
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer
from zope.interface import implementer
from typeclasses.weaviateClient import WeaviateClient as WeaviateClient

import google.generativeai as genai

DEFAULT_LLM_HOST = "https://generativelanguage.googleapis.com"
DEFAULT_LLM_PATH = "/v1beta/models/gemini-pro:generateContent"
DEFAULT_LLM_HEADERS = {"Content-Type": "application/json"}
DEFAULT_LLM_PROMPT_KEYNAME = "prompt"
DEFAULT_LLM_API_TYPE = "gemini"  # or openai
DEFAULT_LLM_REQUEST_BODY = []
DEFAULT_GOOGLE_API_KEY = ""

class GeminiClient:
  """
  A client for communicating with the Google Gemini Pro API.
  """

  def __init__(self, history=[]):

    self.prompt_keyname = getattr(settings, "LLM_PROMPT_KEYNAME", DEFAULT_LLM_PROMPT_KEYNAME)
    self.hostname = getattr(settings, "LLM_HOST", DEFAULT_LLM_HOST)
    self.pathname = getattr(settings, "LLM_PATH", DEFAULT_LLM_PATH)
    self.headers = getattr(settings, "LLM_HEADERS", DEFAULT_LLM_HEADERS)
    self.api_type = getattr(settings, "LLM_API_TYPE", DEFAULT_LLM_API_TYPE)
    self.api_key = getattr(settings, "GOOGLE_API_KEY", DEFAULT_GOOGLE_API_KEY)

    genai.configure(api_key=getattr(settings, "GOOGLE_API_KEY", DEFAULT_GOOGLE_API_KEY))
    self.model = genai.GenerativeModel('gemini-pro')
    self.chat = self._set_history(history)

  def _set_history(self, history):
      """Set the history for the LLM server
          Args:
            history (list): This is the chat history so far, and will be added to the
              prompt in a way suitable for the api."""

      logger.log_info(f"Gemini NPC History: {history}")
      return self.model.start_chat(history=history)


  def _format_request_body(self, prompt):
      """Structure the request body for the LLM server"""
      request_body = self.request_body.copy()
      # TODO: This needs to be updated to use the History JSON format
      prompt = "\n".join(make_iter(prompt))

      request_body[self.prompt_keyname] = prompt

      return request_body

  def _handle_llm_response_body(self, response):
      """Get the response body from the response"""
      d = defer.Deferred()
      response.deliverBody(SimpleResponseReceiver(response.code, d))
      return d

  def _handle_llm_error(self, failure):
      """Correctly handle server connection errors"""
      failure.trap(Exception)
      return (500, failure.getErrorMessage())

  def _get_response_from_llm_server(self, text):
      #history = self.history
      logger.log_info(f"GeminiClient._get_response_from_llm_server({self}, {text})")
      """Call the LLM server and handle the response/failure"""

      logger.log_info(f"User input: {text}")

      response = self.chat.send_message(text)

      logger.log_info(f"Gemini API response: {response}")

      return response.text
      #return d

  @inlineCallbacks
  def get_response(self, text):
      logger.log_info(f"GeminiClient.get_response: {text}")
      """
      Get a response from the LLM server for the given npc.

      Args:
          text (str): The prompt to send to the LLM server.
          
      Returns:
          str: The generated text response. Will return an empty string
              if there is an issue with the server, in which case the
              the caller is expected to handle this gracefully.

      """

      response = yield self._get_response_from_llm_server(text)
      #logger.log_info(self.weaviate_client.create_schema())
      logger.log_info(f"GeminiClient.get_response: {response}")
      return response
