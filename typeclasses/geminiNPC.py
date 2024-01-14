"""
Basic class for NPC that makes use of an LLM (Large Language Model) to generate replies.

It comes with a `talk` command; use `talk npc <something>` to talk to the NPC. The NPC will
respond using the LLM response.

Makes use of the LLMClient for communicating with the server. The NPC will also
echo a 'thinking...' message if the LLM server takes too long to respond.


"""
from datetime import datetime, timezone
from collections import defaultdict
from random import choice

from django.conf import settings
from twisted.internet.defer import inlineCallbacks

from evennia import AttributeProperty, logger
from typeclasses.characters import Character

from typeclasses.geminiClient import GeminiClient

import weaviate
import json

DEFAULT_LLM_REQUEST_BODY = []
DEFAULT_WEAVIATE_URL = ""
DEFAULT_WEAVIATE_KEY = ""

class GeminiNPC(Character):
    """An NPC that uses the Google Gemini API server to generate its responses. If the server is slow, it will
    echo a thinking message to the character while it waits for a response."""

    def at_server_start(self):
      #logger.log_info(f"GeminiNPC.at_server_start: {self}")
      self.geminiClient = GeminiClient()

      self.history = getattr(settings, "DEFAULT_LLM_REQUEST_BODY", DEFAULT_LLM_REQUEST_BODY)

      self.history.append({
            "parts": [
                {"text":f"You are currently located in a room called {self.location}."},
                {"text":f"{self.location.db.desc}"}
            ],
            "role": "user"
      })
      self.history.append({
        "parts": {
          "text":f"think I am located in a room called {self.location}."
        },
        "role": "model"
      })

      memoryArray = [{
          "text":"Here is a query result from your long term memory. Use this to answer my messages."
      }]

      memoryResult = self.query_memories()

      #logger.log_info(f"memoryResult: {memoryResult}")

      for memory in memoryResult['data']['Get']['Memories']:
          #logger.log_info(f"memory: {memory}")
          memoryArray.append({"text": json.dumps(memory)})
      
      memoryArray.append({"text": "End of query result."})

      #logger.log_info(f"memoryArray: {memoryArray}")

      self.history.append({
        "parts": memoryArray,
        "role": "user"
      })

      self.history.append({
        "parts": {
          "text":f"I am grateful for the ability to remember things! I should use the things in my memory to help me respond to users."
        },
        "role": "model"
      })

      self.geminiClient._set_history(self.history)

    # use this to override the prefix per class. Assign an Attribute to override per-instance.
    prompt_prefix = None

    thinking_timeout = AttributeProperty(2, autocreate=False)  # seconds
    thinking_messages = AttributeProperty(
        [
            "{name} thinks about what you said ...",
            "{name} ponders your words ...",
            "{name} ponders ...",
        ],
        autocreate=False,
    )

    max_chat_memory_size = AttributeProperty(25, autocreate=False)
    # this is a store of {character: [chat, chat, ...]}
    chat_memory = AttributeProperty(defaultdict(list))

    def delete_memories(self):
        # Instantiate the client with the auth config
        wClient = weaviate.Client(
            url=getattr(settings, "WEAVIATE_URL", DEFAULT_WEAVIATE_URL),  # Replace w/ your endpoint
            auth_client_secret=weaviate.AuthApiKey(api_key=getattr(settings, "WEAVIATE_KEY", DEFAULT_WEAVIATE_KEY)),  # Replace w/ your Weaviate instance API key
        )
        # TODO: Find out if this works
        response = wClient.schema.delete_class("Memories")
        logger.log_info(f"Deleted Existing Memories: {response}")

    def initialize_memories(self):
        wClient = weaviate.Client(
            url=getattr(settings, "WEAVIATE_URL", DEFAULT_WEAVIATE_URL),  # Replace w/ your endpoint
            auth_client_secret=weaviate.AuthApiKey(api_key=getattr(settings, "WEAVIATE_KEY", DEFAULT_WEAVIATE_KEY)),  # Replace w/ your Weaviate instance API key
        )
        class_obj = {
          "class": "Memories",
          "vectorizer": "text2vec-huggingface",
          "description": "LLM Memories",
          "properties": [
            {
              "dataType": ["text"],
              "description": "Self",
              "name": "self"
            },
            {
              "dataType": ["text"],
              "description": "Text content",
              "name": "text"
            },
            {
              "dataType": ["text"],
              "description": "Origin object",
              "name": "from_obj"
            },
            {
              "dataType": ["date"],
              "description": "Timestamp of the data entry",
              "name": "timestamp"
            }
          ],
          "vectorIndexConfig": {
              "distance": "cosine",
          },
        }
        response = wClient.schema.create_class(class_obj)  # returns null on success
        logger.log_info(f"Create Memories Class: {response}")

    def add_memory(self, text, from_obj):
        response = ""
        try:
          wClient = weaviate.Client(
              url=getattr(settings, "WEAVIATE_URL", DEFAULT_WEAVIATE_URL),  # Replace w/ your endpoint
              auth_client_secret=weaviate.AuthApiKey(api_key=getattr(settings, "WEAVIATE_KEY", DEFAULT_WEAVIATE_KEY)),  # Replace w/ your Weaviate instance API key
          )

          timestamp = datetime.now(timezone.utc).isoformat() or None
          #logger.log_info(f"self.name type: {type(self.name)}")

          if type(from_obj) is Character:
              from_obj = str(from_obj.name)
          elif type(from_obj) is tuple:
              from_obj = str(from_obj[0])

          data_object={
            "self": self.name,
            "text": text,
            "from_obj": from_obj,
            "timestamp": timestamp
          }

          response = wClient.data_object.create(class_name="Memories", data_object=data_object) # returns UUID of the new object

          #logger.log_info(f"Adding Memory: {data_object}")
          logger.log_info(f"Add Memory Response: {response}")
        except Exception as err:
          logger.log_info(f"Error adding memory: {err}")
        return response

    def query_memories(self, text=None, from_obj=None):
        try:
          query_result = {}
          wClient = weaviate.Client(
              url=getattr(settings, "WEAVIATE_URL", DEFAULT_WEAVIATE_URL),  # Replace w/ your endpoint
              auth_client_secret=weaviate.AuthApiKey(api_key=getattr(settings, "WEAVIATE_KEY", DEFAULT_WEAVIATE_KEY)),  # Replace w/ your Weaviate instance API key
          )
          n = 5 # number (int) of memories to return
          certainty = 0.8 # float between 0 and 1
          # TODO: Find a way to handle this dynamically
          moveTo = {
                "concepts": [""],
                "force": 0.85
              } or {} # string of the object to move to TODO define "move to"

          # If there is no input, return the most recent n memories
          if text is None:
            query_result = wClient.query\
              .get("Memories", ["self","text","from_obj", "timestamp"])\
              .with_sort({
                  'path': ['timestamp'],
                  'order': 'desc',
              }).do()
            pass
          # else return the closest n matches by certainty
          else:
            if type(text) is tuple:
                text = text[0]

            near_text_filter = {
              "concepts": [text],
              "certainty": certainty
            }

            # This will need to be dynamically generated eventually
            query_result = wClient.query\
                .get("Memories", ["self","text","from_obj", "timestamp"])\
                .with_near_text(near_text_filter)\
                .with_additional(["certainty"])\
                .with_limit(n)\
                .do()
          #logger.log_info(f"Query Result: {query_result}")
          return query_result
        except Exception as err:
           return err

    @inlineCallbacks
    def at_msg_receive(self, text=None, from_obj=None, **kwargs):
      logger.log_info(f"GeminiNPC.at_msg_receive: {self}, {text}, {from_obj}, {type(from_obj)}, {kwargs}")
      """Called when this NPC is talked to by a character."""
      if from_obj == self or from_obj is None:
        return

      def _respond(response):
        """Async handling of the server response"""
        logger.log_info(f"_respond: {response}")

        if response:
            # remember this response
            self.add_memory(from_obj, text)
        else:
            response = "... I'm sorry, I was distracted. Can you repeat?"

        self.add_memory(self.name, response)
        command, value = response.split(' ', 1)
        if command == 'say':
          result = self.execute_cmd(f"say {value}")
        elif command == 'emote':
          result = self.execute_cmd(f"emote {value}")
        else:
          #from_obj.msg(response)
          result = None

        #result = self.execute_cmd(f"{response}")
        logger.log_info(f"result: {result}")
        #from_obj.msg(response)
      
      memoryArray = []
      memoryResult = self.query_memories()

      #logger.log_info(f"memoryResult: {memoryResult}")

      memoryArray.append({"text": "Recent Memory Result."})
      #builtins.TypeError: 'UnexpectedStatusCodeException' object is not subscriptable
      if (type(memoryResult) is not dict or not memoryResult.get('data')):
        memoryArray.append({"text": f"No memories found. {memoryResult}"})
      else:
        for memory in memoryResult['data']['Get']['Memories']:
            #logger.log_info(f"memory: {memory}")
            memoryArray.append({"text": json.dumps(memory)})
      
      memoryArray.append({"text": "End of Recent Memory Result."})
      memoryArray.append({"text": "Begin Related Context Memory Result."})


      # build the prompt
      memoryResult = self.query_memories(text, from_obj)
      #builtins.TypeError: 'UnexpectedStatusCodeException' object is not subscriptable
      if (type(memoryResult) is not dict or not memoryResult.get('data')):
        memoryArray.append({"text": f"No memories found. {memoryResult}"})
      else:
        # If memoryResult has no data return no memories found
        for memory in memoryResult['data']['Get']['Memories']:
            memoryArray.append({"text": json.dumps(memory)})
      
      memoryArray.append({"text": "End of Related Context Memory Result."})
      memoryArray.append({"text": f"The time is now {datetime.now(timezone.utc).isoformat()}"})
      memoryArray.append({"text": "Respond only with a command word followed by a single line of text."})
      memoryArray.append({"text": f"Your current available commands are: {self.cmdset.current.get_all_cmd_keys_and_aliases()}"})
      content = [f"Your memories are: {memoryArray}"]
      #logger.log_info(f"memoryArray: {memoryArray}")


      content.append(f"Current Event: {text}")
      
      # get the response from the LLM server
      yield self.geminiClient.get_response(content).addCallback(_respond)

