from fastmcp import FastMCP

# mcp server 객체 생성 
mcp = FastMCP(name = "calculator")

# 해당 함수를 MCP 도구로 등록하는 데코레이터 
@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers together"""
    return a * b

if __name__ == "__main__":
    mcp.run()