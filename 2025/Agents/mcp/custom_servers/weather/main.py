from typing import Any

import requests

from mcp.server.fastmcp import FastMCP


####################################################################################
# Temporary monkeypatch which avoids crashing when a POST message is received
# before a connection has been initialized, e.g: after a deployment.
# Source: https://github.com/modelcontextprotocol/python-sdk/issues/423
# pylint: disable-next=protected-access
from mcp.server.session import ServerSession

old__received_request = ServerSession._received_request


async def _received_request(self, *args, **kwargs):
    try:
        return await old__received_request(self, *args, **kwargs)
    except RuntimeError:
        pass


# pylint: disable-next=protected-access
ServerSession._received_request = _received_request
####################################################################################


# Initialize FastMCP server
mcp = FastMCP("weather")


# Adding this decorator allows the function to be used as an MCP tool
@mcp.tool()
def get_weather(latitude: float, longitude: float) -> dict[str, Any]:
    """
    Returns the current weather in a given location. Use your background knowledge to
    guess the latitude and longitude if the user doesn't provide them.

    Example:
    --------
    To get the weather in Montreal, you can use latitude=45.5017 and longitude=-73.5673.

    Parameters:
    -----------
    latitude : float
        The latitude of the location (decimal system).
    longitude : float
        The longitude of the location (decimal system).

    Returns:
    --------
    dict
        The current weather data.

    """
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,showers,snowfall,weather_code,cloud_cover,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m"
    )
    data = response.json()
    return data["current"]


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run()
