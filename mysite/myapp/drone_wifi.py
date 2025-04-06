import time
import subprocess
import re
import pywifi
from pywifi import const


def get_current_ssid():
    """
    使用Windows的netsh命令获取当前连接的SSID
    """
    try:
        output_bytes = subprocess.check_output(["netsh", "wlan", "show", "interfaces"])
        # 解码时忽略非法字节
        output = output_bytes.decode("gbk", errors="ignore")
        match = re.search(r"SSID\s+:\s+(.*)", output)
        if match:
            ssid = match.group(1).strip()
            if ssid:
                return ssid
    except Exception as e:
        print("获取当前SSID失败:", e)
    return None


def is_connected(target_ssid):
    """
    判断当前是否已连接到目标 SSID
    """
    current_ssid = get_current_ssid()
    return current_ssid == target_ssid


def wifi_connect(target_ssid):
    """
    如果当前未连接到目标Wi-Fi，则持续扫描并尝试连接，直到连接成功。
    """
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]

    while True:
        if is_connected(target_ssid):
            print(f"已经连接到 {target_ssid}")
            break
        print(f"未连接到 {target_ssid}，正在尝试连接...")

        # 扫描网络
        iface.scan()
        time.sleep(2)

        networks = iface.scan_results()
        target_found = any(network.ssid == target_ssid for network in networks)
        if not target_found:
            print(f"目标 Wi-Fi {target_ssid} 未发现，等待下一次扫描...")
            time.sleep(10)
            continue

        # 断开当前网络
        iface.disconnect()
        time.sleep(2)

        # 检查是否已有目标 Wi-Fi 的配置
        target_profile = None
        for profile in iface.network_profiles():
            if profile.ssid == target_ssid:
                target_profile = profile
                break

        # 如果没有找到目标配置，则创建一个新的
        if target_profile is None:
            profile = pywifi.Profile()
            profile.ssid = target_ssid
            profile.auth = const.AUTH_ALG_OPEN
            profile.hidden = False
            # 添加新配置
            target_profile = iface.add_network_profile(profile)

        # 发起连接
        iface.connect(target_profile)
        time.sleep(5)

        if iface.status() == const.IFACE_CONNECTED and is_connected(target_ssid):
            print(f"成功连接到 {target_ssid}")
            break
        else:
            print("连接失败，继续尝试...")
        time.sleep(5)


if __name__ == '__main__':
    target_ssid = "TELLO-FDDA9E"
    wifi_connect(target_ssid)

