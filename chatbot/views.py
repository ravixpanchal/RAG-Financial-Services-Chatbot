import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ChatMessage
from .tasks import process_finance_query
from celery.result import AsyncResult

def home(request):
    return render(request, 'chatbot/index.html')

@csrf_exempt
def chat_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('message', '')
            
            if not user_input:
                return JsonResponse({'error': 'No message provided'}, status=400)
                
            # Handle new conversation request
            if user_input.lower().strip() in ['new chat', 'new conversation', 'clear chat', 'restart']:
                ChatMessage.objects.all().delete()
                return JsonResponse({'response': "Chat history has been cleared. Let's start a new conversation!"})
                
            # Retrieve recent chat history for context (last 5 messages)
            recent_msgs = ChatMessage.objects.order_by('-timestamp')[:5]
            chat_history = ""
            for msg in reversed(recent_msgs):
                chat_history += f"User: {msg.user_message}\nAssistant: {msg.bot_response}\n"
            
            system_prompt = data.get('system_prompt', '')
            
            # Dispatch to Celery
            task = process_finance_query.delay(user_input, chat_history, system_prompt)
            
            return JsonResponse({'task_id': task.id, 'status': 'PENDING'})
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_exempt
def task_status_api(request, task_id):
    res = AsyncResult(task_id)
    if res.ready():
        if res.failed():
            return JsonResponse({'status': 'FAILURE', 'error': str(res.result)})
        
        result = res.result
        if isinstance(result, dict) and 'error' in result:
             return JsonResponse({'status': 'FAILURE', 'error': result['error']})
        
        # Ensure result is serializable
        try:
            return JsonResponse({'status': 'SUCCESS', 'result': result})
        except TypeError:
            return JsonResponse({'status': 'SUCCESS', 'result': str(result)})
    else:
        return JsonResponse({'status': res.status})

@csrf_exempt
def rate_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message_id = data.get('message_id')
            rating = data.get('rating') # 1 for up, -1 for down
            
            if not message_id or rating is None:
                return JsonResponse({'error': 'Missing parameters'}, status=400)
                
            msg = ChatMessage.objects.get(id=message_id)
            msg.rating = rating
            msg.save()
            return JsonResponse({'status': 'success'})
        except ChatMessage.DoesNotExist:
            return JsonResponse({'error': 'Message not found'}, status=404)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Method not allowed'}, status=405)
@csrf_exempt
def upload_file_api(request):
    if request.method == 'POST':
        files = request.FILES.getlist('files')
        if not files:
            return JsonResponse({'error': 'No files provided'}, status=400)
            
        try:
            import os
            import uuid
            from django.conf import settings
            from .tasks import index_documents

            # Create temp directory for uploads
            temp_dir = os.path.join(settings.BASE_DIR, 'temp_uploads')
            os.makedirs(temp_dir, exist_ok=True)
            
            file_data_list = []
            for f in files:
                unique_name = f"{uuid.uuid4()}_{f.name}"
                temp_path = os.path.join(temp_dir, unique_name)
                
                with open(temp_path, 'wb+') as destination:
                    for chunk in f.chunks():
                        destination.write(chunk)
                
                file_data_list.append({
                    'name': f.name,
                    'path': temp_path
                })

            # Dispatch background task
            task = index_documents.delay(file_data_list)
            
            return JsonResponse({
                'status': 'success', 
                'task_id': task.id,
                'message': 'Documents are being processed in the background.'
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Method not allowed'}, status=405)
