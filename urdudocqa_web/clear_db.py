#!/usr/bin/env python3
"""
Database cleanup utility for Urdu Document Q&A
Clears vector database to resolve dimension mismatches
"""

import os
import shutils
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_vector_database(persist_dir="chroma_db"):
    """Clear the vector database"""
    try:
        if os.path.exists(persist_dir):
            shutils.rmtree(persist_dir)
            logger.info(f"Vector database cleared: {persist_dir}")
            return True
        else:
            logger.info("No vector database found to clear")
            return True
    except Exception as e:
        logger.error(f"Error clearing vector database: {str(e)}")
        return False

def clear_uploaded_files(data_dir="data"):
    """Clear uploaded PDF files"""
    try:
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info(f"Uploaded files cleared: {data_dir}")
            return True
        else:
            logger.info("No uploaded files found to clear")
            return True
    except Exception as e:
        logger.error(f"Error clearing uploaded files: {str(e)}")
        return False

def main():
    """Main cleanup function"""
    logger.info("Starting database cleanup...")
    
    # Clear vector database
    if clear_vector_database():
        logger.info("✅ Vector database cleared successfully")
    else:
        logger.error("❌ Failed to clear vector database")
        return False
    
    # Ask user if they want to clear uploaded files
    try:
        response = input("Do you want to clear uploaded PDF files? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            if clear_uploaded_files():
                logger.info("✅ Uploaded files cleared successfully")
            else:
                logger.error("❌ Failed to clear uploaded files")
    except KeyboardInterrupt:
        logger.info("Cleanup cancelled by user")
        return False
    
    logger.info("🎉 Cleanup completed! You can now upload your PDF again.")
    return True

if __name__ == "__main__":
    main()
