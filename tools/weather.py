import logging

import requests

from langchain_core.tools import tool


# ============================================================
# Logger
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# 天气工具
# ============================================================

@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气信息。

    包括温度、天气状况和湿度。

    Args:
        city: 要查询天气的城市名称。

    Returns:
        天气查询结果。
    """

    try:

        logger.info(
            "开始查询天气：city=%s",
            city,
        )

        url = (
            f"https://wttr.in/{city}"
            "?format=j1&lang=zh"
        )

        response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        current = data[
            "current_condition"
        ][0]

        current_temp = current[
            "temp_C"
        ]

        current_humidity = current[
            "humidity"
        ]

        current_weather = (
            current[
                "weatherDesc"
            ][0]["value"]
        )

        result = (
            f"温度：{current_temp}\n"
            f"湿度：{current_humidity}\n"
            f"天气：{current_weather}\n"
        )

        logger.info(
            "天气查询成功：city=%s",
            city,
        )

        return result


    except requests.RequestException as e:

        logger.error(
            "天气请求失败：city=%s error=%s",
            city,
            e,
        )

        return (
            "天气查询失败，可能是网络连接问题，"
            "请稍后再试。"
        )


    except (
        KeyError,
        IndexError,
        TypeError,
    ) as e:

        logger.error(
            "天气数据格式异常：city=%s error=%s",
            city,
            e,
        )

        return (
            "天气数据格式异常，"
            "暂时无法获取天气信息。"
        )