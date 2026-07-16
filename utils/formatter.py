import pandas as pd



def to_bold_unicode(text):
    """Convert typical alphanumeric characters into mathematical bold characters to enforce formatting."""
    normal_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    bold_chars = (
        "\U0001D400\U0001D401\U0001D402\U0001D403\U0001D404\U0001D405\U0001D406\U0001D407"
        "\U0001D408\U0001D409\U0001D40A\U0001D40B\U0001D40C\U0001D40D\U0001D40E\U0001D40F"
        "\U0001D410\U0001D411\U0001D412\U0001D413\U0001D414\U0001D415\U0001D416\U0001D417"
        "\U0001D418\U0001D419"
        "\U0001D41A\U0001D41B\U0001D41C\U0001D41D\U0001D41E\U0001D41F\U0001D420\U0001D421"
        "\U0001D422\U0001D423\U0001D424\U0001D425\U0001D426\U0001D427\U0001D428\U0001D429"
        "\U0001D42A\U0001D42B\U0001D42C\U0001D42D\U0001D42E\U0001D42F\U0001D430\U0001D431"
        "\U0001D432\U0001D433"
        "\U0001D7CE\U0001D7CF\U0001D7D0\U0001D7D1\U0001D7D2\U0001D7D3\U0001D7D4\U0001D7D5"
        "\U0001D7D6\U0001D7D7"
    )
    trans = str.maketrans(normal_chars, bold_chars)
    return str(text).translate(trans)



def format_table(df):
    """Utility to convert DataFrame column names to bold, uppercase representations."""
    if df is not None and not df.empty:
        df = df.copy()
        df.columns = [to_bold_unicode(str(col).upper()) for col in df.columns]
    return df