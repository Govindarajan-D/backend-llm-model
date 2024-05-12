import requests
import json
import os
from dotenv import load_dotenv
import asyncio
from azure.eventhub import EventData
from azure.eventhub.aio import EventHubProducerClient
from azure.identity.aio import DefaultAzureCredential

load_dotenv()

news_api_key = os.getenv('NEWS_API_KEY')
EVENT_HUB_FULLY_QUALIFIED_NAMESPACE = os.getenv('EVENT_HUB_FULLY_QUALIFIED_NAMESPACE')
EVENT_HUB_NAME = os.getenv('EVENT_HUB_NAME')

plain_url = 'https://newsdata.io/api/1/news'
params = {'apikey':news_api_key,'qInTitle':'S&P 500','language':'en','category':'business,lifestyle,technology'}

news_response = requests.get(url=plain_url, params=params)

if news_response.ok:
    response_content = json.loads(news_response.content)
    news_data = response_content['results']

select_properties = [{key: item[key] for key in ('article_id','title','link','keywords','description','pubDate','country','category')} for item in news_data]
select_properties = [{"id" if k == "article_id" else k:v for k,v in select_property.items()} for select_property in select_properties]
with open('data.json', 'w') as file:
    json.dump(select_properties, file, indent=4)

credential = DefaultAzureCredential()

async def run():
    # Create a producer client to send messages to the event hub.
    producer = EventHubProducerClient(
        fully_qualified_namespace=EVENT_HUB_FULLY_QUALIFIED_NAMESPACE,
        eventhub_name=EVENT_HUB_NAME,
        credential=credential,
    )
    async with producer:
        # Create a batch.

        for item in select_properties:
            event_data_batch = await producer.create_batch()

            # Add events to the batch.
            event_data_batch.add(EventData(json.dumps(item)))
            # Send the batch of events to the event hub.
            await producer.send_batch(event_data_batch)

        # Close credential when no longer needed.
        await credential.close()

asyncio.run(run())