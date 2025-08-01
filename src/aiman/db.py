"""
MongoDB database operations for conversation management.
Handles user sessions, conversation history, and sliding window memory.
"""
import pymongo
from pymongo import MongoClient
from datetime import datetime, timedelta
import logging
from aiman.config import MONGODB_URI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationDB:
    def __init__(self):
        """Initialize MongoDB connection."""
        try:
            self.client = MongoClient(MONGODB_URI)
            self.db = self.client.aiman_db
            self.conversations = self.db.conversations
            
            # Create indexes for better performance
            self.conversations.create_index([("user_id", 1), ("timestamp", -1)])
            self.conversations.create_index([("timestamp", 1)], expireAfterSeconds=7*24*3600)  # 7 days TTL
            
            logger.info("✅ MongoDB connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    def save_message(self, user_id: str, role: str, message_text: str, metadata: dict = None):
        """
        Save a conversation message to database.
        
        Args:
            user_id: WhatsApp user identifier
            role: 'user' or 'assistant'
            message_text: Message content
            metadata: Additional message metadata
        """
        try:
            message_doc = {
                "user_id": user_id,
                "role": role,
                "text": message_text,
                "timestamp": datetime.utcnow(),
                "metadata": metadata or {}
            }
            
            self.conversations.insert_one(message_doc)
            
            # Implement sliding window - keep only last 30 messages per user
            self._cleanup_old_messages(user_id, limit=30)
            
            logger.info(f"Saved message for user {user_id}: {role}")
            
        except Exception as e:
            logger.error(f"Failed to save message: {str(e)}")
    
    def get_conversation_history(self, user_id: str, limit: int = 15) -> list:
        """
        Get recent conversation history for a user.
        
        Args:
            user_id: WhatsApp user identifier  
            limit: Maximum messages to retrieve
        
        Returns:
            List of conversation messages
        """
        try:
            messages = list(
                self.conversations
                .find({"user_id": user_id})
                .sort("timestamp", -1)  # Latest first
                .limit(limit)
            )
            
            # Reverse to get chronological order
            messages.reverse()
            
            logger.info(f"Retrieved {len(messages)} messages for user {user_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {str(e)}")
            return []
    
    def _cleanup_old_messages(self, user_id: str, limit: int = 30):
        """Clean up old messages beyond the sliding window limit."""
        try:
            # Get messages beyond the limit
            old_messages = list(
                self.conversations
                .find({"user_id": user_id})
                .sort("timestamp", -1)
                .skip(limit)
            )
            
            if old_messages:
                old_ids = [msg["_id"] for msg in old_messages]
                self.conversations.delete_many({"_id": {"$in": old_ids}})
                logger.info(f"Cleaned up {len(old_ids)} old messages for user {user_id}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old messages: {str(e)}")
    
    def get_user_stats(self, user_id: str) -> dict:
        """Get conversation statistics for a user."""
        try:
            total_messages = self.conversations.count_documents({"user_id": user_id})
            user_messages = self.conversations.count_documents({"user_id": user_id, "role": "user"})
            
            # Get first interaction date
            first_message = self.conversations.find_one(
                {"user_id": user_id},
                sort=[("timestamp", 1)]
            )
            
            return {
                "total_messages": total_messages,
                "user_messages": user_messages,
                "first_interaction": first_message.get("timestamp") if first_message else None,
                "is_returning_user": total_messages > 2
            }
            
        except Exception as e:
            logger.error(f"Failed to get user stats: {str(e)}")
            return {"total_messages": 0, "user_messages": 0, "is_returning_user": False}
    
    def cleanup_inactive_users(self, days: int = 7):
        """Remove conversations older than specified days."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            result = self.conversations.delete_many({"timestamp": {"$lt": cutoff_date}})
            logger.info(f"Cleaned up {result.deleted_count} old conversations")
            
        except Exception as e:
            logger.error(f"Failed to cleanup inactive users: {str(e)}")

# Global database instance
db = ConversationDB()

# Convenience functions
def save_user_message(user_id: str, message: str, metadata: dict = None):
    """Save user message to database."""
    db.save_message(user_id, "user", message, metadata)

def save_bot_response(user_id: str, response: str, metadata: dict = None):
    """Save bot response to database."""
    db.save_message(user_id, "assistant", response, metadata)

def get_history(user_id: str, limit: int = 15) -> list:
    """Get conversation history for user."""
    return db.get_conversation_history(user_id, limit)

def get_user_stats(user_id: str) -> dict:
    """Get user conversation statistics."""
    return db.get_user_stats(user_id)
