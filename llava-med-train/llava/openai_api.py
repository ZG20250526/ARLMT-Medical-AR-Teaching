# import openai
import time
import asyncio
from openai import OpenAI

# Fill in your OpenAI setup params here
# openai.api_type = "azure"
# openai.api_key = 'sk-wXA5qvzT8VLMTcomtPc8vMVT5b3C1WbnBbNgK4gd4mYdPtST'
# openai.api_base = 'https://api.fe8.cn/v1'
# openai.api_version = "2023-03-15-preview"
#
DEPLOYMENT_ID="gpt-4o-turbo"

client = OpenAI(api_key='sk-wXA5qvzT8VLMTcomtPc8vMVT5b3C1WbnBbNgK4gd4mYdPtST' ,
                base_url='https://api.fe8.cn/v1')
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system",
#         "content": "介绍一下广州塔"}
#     ]
# )

# print(response.choices[0].message.content)

async def dispatch_openai_requests(
  deployment_id,
  messages_list,
  temperature,
):
    print("massages_list: ", messages_list)
    async_responses = [
        asyncio.create_task(
           client.chat.completions.create(
               model=deployment_id,#"gpt-4-turbo"
               messages=x,
               temperature=temperature
           )
        )

        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


def call_async(samples, wrap_gen_message, print_result=False):
  message_list = []
  for sample in samples:
    input_msg = wrap_gen_message(sample)
    message_list.append(input_msg)

  try:
    predictions = asyncio.run(
      dispatch_openai_requests(
        deployment_id=DEPLOYMENT_ID,
        messages_list=message_list,
        temperature=0.0,
      )
    )
  except Exception as e:
    print(f"Error in call_async: {e}")
    time.sleep(6)
    return []

  results = []
  for sample, prediction in zip(samples, predictions):
    if prediction:
      #if 'content' in prediction['choices'][0]['message']:
      if 'choices' in prediction and 'message' in prediction['choices'][0]:
        sample['result'] = prediction['choices'][0]['message']['content']
        if print_result:
          print(sample['result'])
        results.append(sample)
  return results

# def call_async(samples, wrap_gen_message, print_result=False):
#     message_list = []
#     for sample in samples:
#         input_msg = wrap_gen_message(sample)
#         message_list.append(input_msg)
#
#     try:
#         predictions = asyncio.run(
#             dispatch_openai_requests(
#                 deployment_id=DEPLOYMENT_ID,
#                 messages_list=message_list,
#                 temperature=0.0,
#             )
#         )
#     except Exception as e:
#         # 获取错误的详细信息，包括错误类型、错误消息和堆栈跟踪
#         error_type = type(e).__name__
#         error_message = str(e)
#         stack_trace = ""
#         import traceback
#         stack_trace = traceback.format_exc()
#
#         print(f"Error in call_async at function call_async:\nError Type: {error_type}\nError Message: {error_message}\nStack Trace:\n{stack_trace}")
#         time.sleep(6)
#         return []
#
#     results = []
#     for sample, prediction in zip(samples, predictions):
#         if prediction:
#             # if 'content' in prediction['choices'][0]['message']:
#             if 'choices' in prediction and 'message' in prediction['choices'][0]:
#                 sample['result'] = prediction['choices'][0]['message']['content']
#                 if print_result:
#                     print(sample['result'])
#                 results.append(sample)
#     return results