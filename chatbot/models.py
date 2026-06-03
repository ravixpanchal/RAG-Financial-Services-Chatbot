from django.db import models

class ChatMessage(models.Model):
    user_message = models.TextField()
    bot_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    rating = models.IntegerField(null=True, blank=True) # 1 for up, -1 for down

    def __str__(self):
        return f"User: {self.user_message[:50]}..."
