import time
import subprocess
import re
import pywifi
import logging
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
        logging.info("获取当前SSID失败:", e)
        print("获取当前SSID失败:", e)
    return None


def get_wifi_signal_strength():
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",  # 忽略无法解码的字符
            check=True
        )
        output = result.stdout
        if output:
            for line in output.split("\n"):
                if "信号" in line or "Signal" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        return parts[1].strip()
    except Exception as e:
        print(f"Error: {e}")
    return "N/A"

def is_connected(target_ssid):
    """
    判断当前是否已连接到目标 SSID
    """
    current_ssid = get_current_ssid()
    logging.info(f"当前SSID: {current_ssid}")
    # print(f"当前SSID: {current_ssid}")
    return current_ssid == target_ssid


def wifi_connect(target_ssid, max_retries=3):
    """
    如果当前未连接到目标Wi-Fi，则持续扫描并尝试连接，直到连接成功。
    """
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]

    retry_count = 0
    while retry_count < max_retries:
        if is_connected(target_ssid):
            logging.info(f"已经连接到 {target_ssid}")
            # print(f"已经连接到 {target_ssid}")
            return {'status': 1, 'message': f'已连接目标WiFi: {target_ssid}'}

        logging.info(f"未连接到 {target_ssid}，正在尝试连接...")
        # print(f"未连接到 {target_ssid}，正在尝试连接...")

        try:
            # 扫描网络
            iface.scan()
            time.sleep(2)
            networks = iface.scan_results()
        except Exception as e:
            logging.error(f"扫描网络失败: {e}")
            return {'status': 0, 'message': f"扫描网络失败"}

        target_found = any(network.ssid == target_ssid for network in networks)
        if not target_found:
            logging.info(f"目标 Wi-Fi {target_ssid} 未发现，等待下一次扫描...")
            # print(f"目标 Wi-Fi {target_ssid} 未发现，等待下一次扫描...")
            time.sleep(5)
            retry_count += 1
            continue

        try:
            # 断开当前网络
            iface.disconnect()
            time.sleep(2)
        except Exception as e:
            logging.error(f"断开当前网络失败: {e}")
            return {'status': 0, 'message': "断开当前网络失败"}

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
            try:
                target_profile = iface.add_network_profile(profile)
            except Exception as e:
                logging.error(f"添加新配置失败: {e}")
                return {'status': 0, 'message': "添加新配置失败"}

        try:
            # 发起连接
            iface.connect(target_profile)
            time.sleep(5)
        except Exception as e:
            logging.error(f"连接到 {target_ssid} 时发生错误: {e}")
            return {'status': 0, 'message': f"连接到 {target_ssid} 失败"}

        if iface.status() == const.IFACE_CONNECTED and is_connected(target_ssid):
            logging.info(f"成功连接到 {target_ssid}")
            # print(f"成功连接到 {target_ssid}")
            return {'status': 1, 'message': f'成功连接到目标WiFi: {target_ssid}'}
            # break
        else:
            logging.info("连接失败，继续尝试...")
            # print("连接失败，继续尝试...")
            retry_count += 1

    logging.error(f"连接到 {target_ssid} 失败，达到最大重试次数")
    # print(f"连接到 {target_ssid} 失败，达到最大重试次数")
    return {'status': 0, 'message': f"连接到 {target_ssid} 失败，达到最大重试次数"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    target_ssid = "TELLO-FDDA9E"
    wifi_connect(target_ssid)
    get_wifi_signal_strength()