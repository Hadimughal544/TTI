import motor.motor_asyncio
import asyncio

async def test():
    uri = "mongodb+srv://corenexinnovation_db_user:GNJYRPBTEFYo1uKN@cluster0.9fbezbn.mongodb.net/corenex_db?retryWrites=true&w=majority&appName=Cluster0"
    client = motor.motor_asyncio.AsyncIOMotorClient(uri)
    try:
        # The ismaster command is cheap and does not require auth.
        await client.admin.command('ping')
        print("MongoDB Connection Successful")
    except Exception as e:
        print(f"MongoDB Connection Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test())
