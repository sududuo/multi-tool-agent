from langchain_core.messages import HumanMessage
from agent.graph import app
import logging

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - "
        "%(levelname)s - "
        "%(name)s - "
        "%(message)s"
    ),
)

def main():

    thread_id = input(
        "请输入thread_id："
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    while True:

        question = input(
            "你："
        )

        if question.lower() in [
            "q",
            "quit",
        ]:
            break

        final_result = app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=question
                    )
                ]
            },
            config=config,
        )

        print(
            final_result["messages"][-1].content
        )


if __name__ == "__main__":
    main()