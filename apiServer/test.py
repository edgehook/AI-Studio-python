import asyncio

async def coroutine1():
    print("协程1开始")
    await asyncio.sleep(2)
    print("协程1结束")
    return 1

async def coroutine2():
    print("协程2开始")
    await asyncio.sleep(3)
    print("协程2结束")
    return 2

async def main():
    results = await asyncio.gather(coroutine1(), coroutine2())
    print("所有协程结果:", results)
    
def run():
    asyncio.run(main())
