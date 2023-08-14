import asyncio
import os
import random

from lxml import etree

from aiohttp import ClientSession
import requests
import aiofiles
from bs4 import BeautifulSoup


def get_page_source(url):
    """
    获取页面源码的方法
    :param url: 传入的url
    :return: 返回的是页面的源码
    """
    # 建立随机headers
    user_agent_list = [
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.67 Safari/537.36"
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36"
        "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2049.0 Safari/537.36"
        "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:22.0) Gecko/20130328 Firefox/22.0"
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1464.0 Safari/537.36"
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.1623.0 Safari/537.36"
        "Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1667.0 Safari/537.36"
        "Mozilla/5.0 (X11; NetBSD) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.116 Safari/537.36"
        "Mozilla/5.0 (Windows NT 5.0; rv:21.0) Gecko/20100101 Firefox/21.0"
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1664.3 Safari/537.36"
        "Opera/9.80 (Windows NT 5.1; U; cs) Presto/2.7.62 Version/11.01"
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1944.0 Safari/537.36"
        "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:21.0) Gecko/20130331 Firefox/21.0"
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36"
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.67 Safari/537.36"
        "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.93 Safari/537.36"
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.67 Safari/537.36"
        "Mozilla/5.0 (X11; OpenBSD i386) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/4E423F"
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.93 Safari/537.36"
    ]
    headers = {"user-agent": random.choice(user_agent_list)}
    res = requests.get(url, headers=headers)
    data = res.content.decode()
    return data


def parse_page_source(html):
    """
    对页面进行解析,得到我们每一个章节的url
    :param html: 传入的页面源码
    :return: 返回的是一个带有所有章节url的集合
    """
    book_list = []
    soup = BeautifulSoup(html, 'html.parser')
    # a_list = soup.find_all('div', attrs={'class': 'mulu-list quanji'})
    a_list = soup.find_all('div', attrs={'class': 'mulu-list'})
    # a_list = soup.find_all('div', attrs={'class': 'mulu-list-2'})
    for a in a_list:
        a_list = a.find_all('a')
        for href in a_list:
            chapter_url = href['href']
            book_list.append(chapter_url)
    return book_list


def get_book_name(chapter_url):
    """
    得到名称,为了后续下载好分辨
    :param chapter_url:
    :return:
    """
    book_chapter_name = chapter_url.split('/')[-2]
    id = chapter_url.split('/')[-1][:-5]
    return book_chapter_name, id


async def aio_download_one_content(chapter_url, single):
    """
    下载一个章节内容
    :param chapter_url: 传入得到的章节url
    :param single: 使用async with single就是500个并发
    :return:
    """
    c_name, c_id = get_book_name(chapter_url)
    for i in range(10):
        try:
            async with single:
                async with ClientSession() as session:
                    async with session.get(chapter_url) as res:
                        # 得到章节内容的页面的源码
                        page_source = await res.content.read()
                        tree = etree.HTML(page_source)
                        # 章节名称
                        base_title = tree.xpath('//h1/text()')[0]
                        if (':' in base_title):
                            number = base_title.split(':')[0]
                            con = base_title.split(':')[1]
                            title = number + con
                        else:
                            title = base_title
                        # 章节内容
                        content = tree.xpath('//div[@class="neirong"]/p/text()')
                        # if content[0][0]=="第":
                        #     del content[0]
                        chapter_content = '\n'.join(content).replace(u'\xa0', '')
                        if not os.path.exists(f'{book_name}'):
                            os.makedirs(f'{book_name}')
                        title = title.replace('*', '').replace('/', '').replace('?', '')
                        # if not os.path.exists(f'{book_name}/{c_name}'):
                        #     os.makedirs(f'{book_name}/{c_name}')
                        # async with aiofiles.open(f'{book_name}/{c_name}/{c_id}{title}.txt', mode="w",
                        #                          encoding='utf-8') as f:
                        async with aiofiles.open(f'{book_name}/{c_id}{title}.txt', mode="w",
                                                 encoding='utf-8') as f:
                            await f.write(chapter_content)
                        print(chapter_url, "下载完毕!")
                        return ""
        except Exception as e:
            print(e)
            print(chapter_url, "下载失败!, 重新下载. ")
    return chapter_url


async def aio_download(url_list):
    # 创建一个任务列表
    tasks = []
    # 设置最多500个任务并行运作
    semaphore = asyncio.Semaphore(500)
    for h in url_list:
        tasks.append(asyncio.create_task(aio_download_one_content(h, semaphore)))
    await asyncio.wait(tasks)


if __name__ == '__main__':
    # url = 'https://www.51shucheng.net/daomu/guichuideng'
    # url = 'https://www.51shucheng.net/wangluo/doupocangqiong'
    # url='https://www.51shucheng.net/wangluo/zhuixu'
    # url='https://www.51shucheng.net/wangluo/xueyinglingzhu'
    # url='https://www.51shucheng.net/wangluo/dafengdagengren'
    # url='https://www.51shucheng.net/xuanhuan/guimizhizhu'
    # url='https://www.51shucheng.net/wangluo/woyouyizuokongbuwu'
    # url='https://www.51shucheng.net/wangluo/huaqiangu'
    # url='https://www.51shucheng.net/wangluo/quanzhifashi'
    # url='https://www.51shucheng.net/wangluo/xiuzhenliaotianqun'
    url='https://www.51shucheng.net/wangluo/xuezhonghandaoxing'
    # url='https://www.51shucheng.net/wangluo/huangjintong'
    # url = 'https://www.51shucheng.net/wangluo/quanzhigaoshou'
    # url='https://www.51shucheng.net/wangluo/douluodalu'
    # book_name = '鬼吹灯'
    # book_name = '斗破苍穹'
    # book_name = '赘婿'
    # book_name= '雪鹰领主'
    # book_name='大奉打更人'
    # book_name='诡秘之主'
    # book_name='我有一座冒险屋'
    # book_name='花千骨'
    # book_name='全职法师'
    # book_name='修真聊天群'
    book_name='雪中悍刀行'
    # book_name='黄金瞳'
    # book_name = '全职高手'
    # book_name='斗罗大陆'
    source = get_page_source(url)
    href_list = parse_page_source(source)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(aio_download(href_list))
