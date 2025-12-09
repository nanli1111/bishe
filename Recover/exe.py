import subprocess
import sys
import time
import os
import datetime

def setup_log_dir():
    """创建一个按时间戳命名的日志目录，方便管理"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = "logs"
    current_log_dir = os.path.join(log_root, timestamp)
    
    if not os.path.exists(current_log_dir):
        os.makedirs(current_log_dir)
    
    print(f"📂 本次运行的日志将保存在: {os.path.abspath(current_log_dir)}\n")
    return current_log_dir

def get_utf8_env():
    """创建一个强制使用 UTF-8 的环境变量副本"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env

def run_process_with_logger(cmd, log_file_path):
    """
    启动进程，智能处理输出：
    - 控制台：显示所有内容（包括进度条动画）
    - 日志文件：过滤掉进度条，只保留关键信息（如 Epoch Summary, Print 语句）
    """
    last_was_progress = False

    with open(log_file_path, "a", encoding="utf-8") as f:
        # 记录开始时间
        start_msg = f"[{datetime.datetime.now()}] 开始执行: {' '.join(cmd)}\n" + "-"*50 + "\n"
        f.write(start_msg)
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True, 
            bufsize=1,
            encoding='utf-8',
            errors='replace',
            env=get_utf8_env() 
        )

        while True:
            try:
                line = process.stdout.readline()
            except UnicodeDecodeError:
                continue 
                
            if not line and process.poll() is not None:
                break
            
            if line:
                # --- 核心修改：判断是否为进度条 ---
                # tqdm 的特征通常包含百分比和速度，或者特定的 Epoch 进度格式
                # 你的进度条例子: "Epoch 1/500:   0%|          | 0/5625 [00:00<?, ?it/s, Loss=0.16643]"
                is_progress_bar = ("%" in line) and (("it/s" in line) or ("s/it" in line) or ("|" in line))

                # 1. 控制台输出逻辑 (保持动画效果)
                if is_progress_bar:
                    print(f"\r{line.strip()}", end="", flush=True)
                    last_was_progress = True
                else:
                    if last_was_progress:
                        print() 
                    print(line, end="", flush=True)
                    last_was_progress = False
                
                # 2. 日志文件写入逻辑 (过滤掉进度条)
                # 只有当这一行 **不是** 进度条时，才写入文件
                if not is_progress_bar:
                    f.write(line)
                    f.flush()

        if last_was_progress:
            print()

        end_msg = "\n" + "-"*50 + f"\n[{datetime.datetime.now()}] 执行结束，返回码: {process.returncode}\n\n"
        f.write(end_msg)
        
        return process.returncode

def dry_run_check(tasks, timeout=3):
    python_exe = sys.executable
    print("=" * 40)
    print(f"🧪 开始冒烟测试 (每个脚本试运行 {timeout} 秒)")
    print("=" * 40)

    all_passed = True

    for i, task in enumerate(tasks, 1):
        if isinstance(task, str):
            cmd = [python_exe, task]
            script_name = task
        else:
            cmd = [python_exe] + task
            script_name = task[0]

        if not os.path.exists(script_name):
            print(f"[{i}/{len(tasks)}] ❌ 文件缺失：{script_name}")
            all_passed = False
            continue

        filename = os.path.basename(script_name)
        print(f"[{i}/{len(tasks)}] ⏳ 试启动: {filename} ...", end="", flush=True)

        try:
            # 【修复点 2】测试运行时也必须强制 UTF-8，否则带 emoji 的脚本一启动就挂
            proc = subprocess.Popen(
                cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE,
                env=get_utf8_env()
            )
            try:
                proc.wait(timeout=timeout)
                if proc.returncode == 0:
                    print(" ✅ (快速完成)")
                else:
                    _, stderr = proc.communicate()
                    print(f" ❌ (报错, Code: {proc.returncode})")
                    if stderr:
                        err_msg = stderr.decode('utf-8', errors='replace').strip()
                        print(f"    错误: {err_msg[:300]}...") 
                    all_passed = False
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait()
                print(" ✅ (启动正常)")

        except Exception as e:
            print(f" ❌ (无法启动: {e})")
            all_passed = False

    print("-" * 40)
    return all_passed

def run_pipeline(tasks):
    python_exe = sys.executable
    log_dir = setup_log_dir()
    
    total_start = time.time()
    
    print("\n" + "=" * 30)
    print("🚀 开始正式批量执行任务")
    print("=" * 30)

    for i, task in enumerate(tasks, 1):
        if isinstance(task, str):
            cmd = [python_exe, task]
            script_name = task
        else:
            cmd = [python_exe] + task
            script_name = task[0]

        base_name = os.path.basename(script_name).replace(".py", "")
        log_filename = f"{i:02d}_{base_name}.log"
        log_path = os.path.join(log_dir, log_filename)

        print(f"[{i}/{len(tasks)}] ▶️ 正在运行: {os.path.basename(script_name)}")
        print(f"    📝 日志文件: {log_path}")
        
        start_time = time.time()

        try:
            return_code = run_process_with_logger(cmd, log_path)
            
            elapsed = time.time() - start_time
            
            if return_code == 0:
                print(f"   ✅ 完成。耗时: {elapsed:.2f}秒\n")
            else:
                print(f"   ❌ 失败！返回码: {return_code} (请查看日志详情)")
                print("🚨 任务队列已终止。")
                sys.exit(1)

        except KeyboardInterrupt:
            print("\n🛑 用户强制停止。")
            sys.exit(1)
        except Exception as e:
            print(f"   ❌ 发生异常: {e}")
            sys.exit(1)

    total_time = time.time() - total_start
    print("=" * 30)
    print(f"🏁 所有流程结束。总耗时: {total_time:.2f}秒")
    print(f"📂 全部日志保存在: {os.path.abspath(log_dir)}")

if __name__ == "__main__":
    my_tasks = [
        r"F:\LJN\bishe\bishe\IS2B\rIS2B_nakagmi_resnet_adjust\train_scope.py",
        r"F:\LJN\bishe\bishe\IS2B\rIS2B_nakagmi_resnet_adjust\train_crum_andlarge.py",
        r"F:\LJN\bishe\bishe\IS2B\rIS2B_nakagmi_resnet_adjust\test_ber.py"
    ]

    # 1. 先进行冒烟测试 (10秒足够了)
    if dry_run_check(my_tasks, timeout=10):
        print("🎉 测试通过！准备开始正式运行...")
        time.sleep(2)
        run_pipeline(my_tasks)
    else:
        print("🚫 测试未通过，请检查上述错误。正式运行已取消。")