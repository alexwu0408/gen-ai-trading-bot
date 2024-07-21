import openai
from autogen import ConversableAgent
import pylint.lint
from line_profiler import LineProfiler
import psutil
import tracemalloc
import time


OPENAI_API_KEY = "YOUR_API_KEY"
openai.api_key = OPENAI_API_KEY

llm_config = {
    "api_key": OPENAI_API_KEY,
    "model": "gpt-3.5-turbo"
}


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define the Classes for the Agents:
# Reuqest Manager Agent, Planner Agent, Engineer Agent, Template Agent, Test & Modifer Agent, Code & Review Agent
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# RDefine equest Manager Agent Class
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CustomRequestManagerAgent(ConversableAgent):
    def __init__(self, name, system_message, llm_config, human_input_mode):
        super().__init__(name, system_message, llm_config, human_input_mode)
        self.user_input = {}
        self.confirmed_request = {}

    def process_request(self, user_input):
        # Process user input and ensure required parameters are included
        required_params = ["strategy", "symbol", "action", "size", "maxtime"]
        missing_params = [param for param in required_params if param not in user_input]
        if missing_params:
            return f"Missing parameters: {', '.join(missing_params)}. Please provide them."
        
        self.user_input = user_input
        return f"Received user input: {user_input}. Please confirm if this is correct."

    def confirm_request(self, confirmation):
        if confirmation.lower() in ["yes", "y"]:
            self.confirmed_request = self.user_input
            return f"Confirmed request: {self.confirmed_request}. Preparing to process."
        else:
            return "Please provide the correct financial algorithm parameters."



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define PlannerAgent Class
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CustomPlannerAgent(ConversableAgent):
    def __init__(self, name, system_message, llm_config, human_input_mode):
        super().__init__(name, system_message, llm_config, human_input_mode)
        self.best_algorithm = {}

    def optimize_strategy(self, algorithm_request):
        # Simulate back-and-forth discussion, simplified to a fixed number of iterations
        for _ in range(3):  # Assume 3 rounds of discussion
            optimized_strategy = algorithm_request.copy()
            optimized_strategy["size"] = int(algorithm_request["size"] * 1.1)  # Example optimization
            optimized_strategy["maxtime"] = int(algorithm_request["maxtime"] * 0.9)  # Example optimization
            algorithm_request = optimized_strategy  # Simulate the discussion process
        self.best_algorithm = optimized_strategy
        return self.best_algorithm




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Engineer Agent Class
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CustomEngineerAgent(ConversableAgent):
    def __init__(self, name, system_message, llm_config, human_input_mode):
        super().__init__(name, system_message, llm_config, human_input_mode)

    def generate_code(self, algorithm):
        # Generate Python code based on the best algorithm
        code = f"""
        import sys
        import time
        import datetime
        import uuid
        import random
        import logging
        from time import sleep
        import pandas as pd
        import argparse
        from Management import Management

        sys.path.append('../')
        format = "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s() line: %(lineno)d: %(message)s"

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            level=logging.INFO)


        # ******** Team I Bot (InfluxDB) ********
        class ExecutionBot(Management):

            def __init__(self, strategy, starting_money,
                         market_event_securities, market_event_queue, securities,
                         host=None, bot_id=None):

                super(ExecutionBot, self).__init__(strategy, starting_money,
                                                   market_event_securities, market_event_queue, securities,
                                                   host, bot_id)

                self.stat = dict()

                self.start()

                self.penalty = .125

                sleep(10)

            def start_task(self, sym, action, size):

                self.stat = {{
                    'strategy': self.strategy,
                    'sym': sym,
                    'action': action,
                    'qty_target': size,
                    'bp': self.mid_market[sym],
                    'vwap': self.vwap[sym]
                }}

            def task_complete(self, pv, qty, time_t, slices):
                self.stat['pv'] = pv
                self.stat['qty'] = qty
                self.stat['time_t'] = time_t
                self.stat['slices'] = slices
                self.stop(self.stat, log=True)

            def aggressive_orders(self, qty, action, exec_t=2, log=True):
                sym = self.securities[0]

                book_side = 'Ask' if action == 'buy' else 'Bid'
                side = 'B' if action == 'buy' else 'S'

                benchmark_price = self.mid_market[sym]
                benchmark_vwap = self.vwap[sym]

                qty_target = qty

                t_start = time.time()

                pv = 0

                while qty > 0 and time.time() - t_start < exec_t:

                    book_levels = self.market_event_queue.copy()

                    size = 0
                    order_prices = []
                    order_qty = []

                    while size < qty and len(book_levels) > 0:
                        level = book_levels.pop(0)
                        try:
                            level_size = self.market_dict[sym][level][book_side + 'Size']
                            level_price = self.market_dict[sym][level][book_side + 'Price']
                            print(
                                f'level is {level}, size in this level is {{level_size}}, price in this level is {{level_price}}')
                            size_level = min(
                                qty-size, self.market_dict[sym][level][book_side + 'Size'])
                            size += int(size_level)

                            order_prices.append(
                                self.market_dict[sym][level][book_side + 'Price'])
                            order_qty.append(size_level)
                            print(f'pty is {{qty}}, size_leve is {{size_level}}')
                        except:
                            pass

                    print(order_prices)
                    print(order_qty)

                    orders = []
                    for p, q in zip(order_prices, order_qty):
                        order = {{'symb': sym,
                                 'price': p,
                                 'origQty': q,
                                 'status': "A",
                                 'remainingQty': q,
                                 'action': "A",
                                 'side': side,
                                 'FOK': 0,
                                 'AON': 0,
                                 'strategy': self.strategy,
                                 'orderNo': self.internalID
                                 }}

                        self.send_order(order)
                        logging.info(f"Aggressive order sent: \n"
                                     f"\t {{order['symb']}}: "
                                     f"{{order['orderNo']}} | "
                                     f"{{order['side']}} | "
                                     f"{{order['origQty']}} | "
                                     f"{{order['remainingQty']}} | "
                                     f"{{order['price']}}")

                        orders.append(order)
                        self.internalID += 1

                    qty = 0

                    for order in orders:
                        in_id = order["orderNo"]

                        if in_id in self.inIds_to_orders_confirmed:
                            order = self.inIds_to_orders_confirmed[in_id]
                            order['orderNo'] = self.inIds_to_exIds[in_id]

                            self.cancel_order(order)
                            self.logger.info(f"Cancelled order: \n"
                                             f"\t {{order['symb']}}: "
                                             f"{{order['orderNo']}} | "
                                             f"{{order['side']}} | "
                                             f"{{order['origQty']}} | "
                                             f"{{order['remainingQty']}} | "
                                             f"{{order['price']}}")

                            qty += order['remainingQty']

                            pv += order['price'] * \
                                (order['origQty'] - order['remainingQty'])
                        else:
                            self.logger.info(f"Fully filled aggressive order: \n"
                                             f"\t {{order['symb']}}: "
                                             f"{{order['orderNo']}} | "
                                             f"{{order['side']}} | "
                                             f"{{order['remainingQty']}} | "
                                             f"{{order['price']}}")

                            pv += order['price'] * order['origQty']

                try:
                    cost_qty = pv / (qty_target - qty) - benchmark_price*1.
                except:
                    cost_qty = 999.99
                    benchmark_price = 999.99
                if action == 'buy':
                    cost_qty *= -1

                logging.info(f'\n\t Aggressive order: {action} {qty_target-qty} {sym} given {min(time.time() - t_start, exec_t)} seconds: \n'
                             f'\t Transaction cost: {cost_qty} per share\n'
                             f'\t Benchmark price {benchmark_price}\n'
                             f'\t Benchmark VWAP: {benchmark_vwap}')

                penalty, pv_final = self.final_liquidation(qty, action)

                cost_qty = (pv + pv_final) / qty_target - benchmark_price
                if action == 'buy':
                    cost_qty *= -1

                return pv, qty

            def twap_orders(self, qty, action, n_slices, exec_t=3.0):

                sym = self.securities[0]

                book_side = 'Ask' if action == 'buy' else 'Bid'
                side = 'B' if action == 'buy' else 'S'

                benchmark_price = self.mid_market[sym]
                benchmark_vwap = self.vwap[sym]

                pre_vwap = benchmark_vwap

                qty_target = qty

                t_start = time.time()

                max_time = exec_t * n_slices

                pv = 0

                qty_slice = 0
                for i in range(n_slices):
                    if qty <= 0:
                        break

                    if action == 'buy':
                        expand_rate = 1.2 if self.vwap[sym] - pre_vwap < 0 else 1
                    else:
                        expand_rate = 1 if self.vwap[sym] - pre_vwap < 0 else 1.2

                    print(f'pre_vwap: {pre_vwap}, current vwap: {self.vwap[sym]}')
                    print(f'expand_rate is {expand_rate}')

                    pre_vwap = self.vwap[sym]
                    target_q = int(qty / (n_slices - i) * expand_rate)

                    qty_slice = 0

                    book_levels = self.market_event_queue.copy()
                    size = 0
                    order_prices = []
                    order_qty = []

                    while size < target_q and len(book_levels) > 0:
                        level = book_levels.pop(0)
                        try:
                            level_size = self.market_dict[sym][level][book_side + 'Size']
                            level_price = self.market_dict[sym][level][book_side + 'Price']
                            print(
                                f'level is {level}, size in this level is {{level_size}}, price in this level is {{level_price}}')

                            size_level = min(
                                target_q - size, self.market_dict[sym][level][book_side + 'Size'])
                            size += int(size_level)

                            order_prices.append(
                                self.market_dict[sym][level][book_side + 'Price'])
                            order_qty.append(size_level)
                        except Exception:
                            print(f'{sym} dont have level {level} price')
                            pass

                    print(order_prices)
                    print(order_qty)

                    orders = []
                    for p, q in zip(order_prices, order_qty):
                        order = {{'symb': sym,
                                 'price': p,
                                 'origQty': q,
                                 'status': "A",
                                 'remainingQty': q,
                                 'action': "A",
                                 'side': side,
                                 'FOK': 0,
                                 'AON': 0,
                                 'strategy': self.strategy,
                                 'orderNo': self.internalID
                                 }}

                        self.send_order(order)
                        logging.info(f"Slice {{i+1}} - twap order sent: \n"
                                     f"\t {{order['symb']}}: "
                                     f"{{order['orderNo']}} | "
                                     f"{{order['side']}} | "
                                     f"{{order['origQty']}} | "
                                     f"{{order['remainingQty']}} | "
                                     f"{{order['price']}}")

                        orders.append(order)
                        self.internalID += 1

                    for order in orders:
                        in_id = order["orderNo"]

                        if in_id in self.inIds_to_orders_confirmed:
                            order = self.inIds_to_orders_confirmed[in_id]
                            order['orderNo'] = self.inIds_to_exIds[in_id]

                            self.cancel_order(order)
                            self.logger.info(f"Cancelled limit order {{order['remainingQty']}} out of {{order['origQty']}}: \n"
                                             f"\t {{order['symb']}}: "
                                             f"{{order['orderNo']}} | "
                                             f"{{order['side']}} | "
                                             f"{{order['remainingQty']}} | "
                                             f"{{order['price']}}")

                            qty_slice += order['remainingQty']

                            pv += order['price'] * \
                                (order['origQty'] - order['remainingQty'])
                        else:
                            self.logger.info(f"Fully filled limit order: \n"
                                             f"\t {{order['symb']}}: "
                                             f"{{order['orderNo']}} | "
                                             f"{{order['side']}} | "
                                             f"{{order['remainingQty']}} | "
                                             f"{{order['price']}}")

                            pv += order['price'] * order['origQty']

                    qty -= size - qty_slice
                    qty_slice += target_q - size

                    if max_time + t_start - time.time() < 1 and qty > 0:
                        print(f'Emergency, go aggressive. Remain qty = {{qty}}')
                        pv_slice, qty_slice = self.aggressive_orders(qty, action)
                        pv += pv_slice
                        break

                try:
                    cost_qty = pv / (qty_target - qty_slice) - benchmark_price * 1.
                except:
                    cost_qty = 999.99
                    benchmark_price = 999.99

                if action == 'buy':
                    cost_qty *= -1

                logging.info(f'\n\t Slicing order: {{action}} {{qty_target-qty_slice}} {{sym}}\n'
                             f'\t Given {{n_slices}} slices per {{exec_t}} seconds: \n'
                             f'\t Transaction cost: {{cost_qty}} per share\n'
                             f'\t Benchmark price: {{benchmark_price}}\n'
                             f'\t Benchmark VWAP: {{benchmark_vwap}}')

                penalty, pv_final = self.final_liquidation(qty_slice, action)

                cost_qty = (pv + pv_final) / qty_target - benchmark_price
                if action == 'buy':
                    cost_qty *= -1

                return pv, qty

            def final_liquidation(self, qty, action, exec_t=30):
                penalty = 0
                pv_final = 0

                if qty > 0:
                    pv_final, _ = self.aggressive_orders(qty, action, exec_t)
                    penalty = self.penalty * qty

                return penalty, pv_final


        if __name__ == "__main__":
            myargparser = argparse.ArgumentParser()
            myargparser.add_argument('--strategy', type=str,
                                     const="{algorithm['strategy']}", nargs='?', default="{algorithm['strategy']}")
            myargparser.add_argument('--symbol', type=str,
                                     const="{algorithm['symbol']}", nargs='?', default="{algorithm['symbol']}")
            myargparser.add_argument('--action', type=str,
                                     const="{algorithm['action']}", nargs='?', default="{algorithm['action']}")
            myargparser.add_argument(
                '--size', type=int, const={algorithm['size']}, nargs='?', default={algorithm['size']})
            myargparser.add_argument('--maxtime', type=int,
                                     const={algorithm['maxtime']}, nargs='?', default={algorithm['maxtime']})
            myargparser.add_argument('--username', type=str, default='test')
            myargparser.add_argument('--password', type=str, default='test')
            myargparser.add_argument('--bot_id', type=str,
                                     const='text', nargs='?', default='text')
            args = myargparser.parse_args()

            market_event_securities = [args.symbol]
            market_event_queue = ["L1", "L2", "L3", "L4", "L5"]
            securities = market_event_securities
            host = "localhost"
            strategy = args.strategy
            bot_id = args.bot_id
            starting_money = 1000000000.0

            start_t = time.time()
            exec_bot = ExecutionBot(strategy, starting_money, market_event_securities,
                                    market_event_queue, securities, host, bot_id)
            exec_bot.start_task(args.symbol, args.action, args.size)

            pv, qty, num_slices = 0, 0, 10
            pv, qty = exec_bot.twap_orders(
                args.size, args.action, num_slices, int(args.maxtime/num_slices))

            end_t = time.time()
            exec_bot.task_complete(pv, qty, end_t-start_t, num_slices)
        """
        return code
    

    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define Template Agent Class
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CustomTemplateAgent(ConversableAgent):
    def __init__(self, name, system_message, llm_config, human_input_mode):
        super().__init__(name, system_message, llm_config, human_input_mode)
        self.code = ""
    
    def review_code(self, code):
        # Check if the code meets the required template
        required_imports = ["import sys", "import time", "import datetime", "import uuid", "import random", "import logging", "from time import sleep", "import pandas as pd", "import argparse", "from Management import Management"]
        for imp in required_imports:
            if imp not in code:
                return "Code does not meet the template requirements. Missing import: " + imp
        if "class ExecutionBot(Management):" not in code:
            return "Code does not meet the template requirements. Missing class definition: ExecutionBot"
        return "Code meets the template requirements."
    
    def compile_code(self, code):
        # Simulate compilation process, here should include actual compilation logic
        try:
            exec(code)
        except Exception as e:
            return f"Compilation failed. Error: {e}"
        return "Compilation successful. Code is ready for testing."
    
    def save_code(self, code, filename="execution_bot.py"):
        # Save the code to a .py file
        with open(filename, "w") as file:
            file.write(code)
        return filename



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Tester & Modifier Agent
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CustomTesterModifierAgent(ConversableAgent):
    def __init__(self, name, system_message, llm_config, human_input_mode):
        super().__init__(name, system_message, llm_config, human_input_mode)
        self.code_file = ""

    def test_code(self, code_file):
        # Simulate code testing using various tools
        test_results = {
            "runtime_execution": self.measure_runtime(code_file),
            "memory_usage": self.measure_memory_usage(code_file),
            "CPU_utilization": self.measure_cpu_utilization(code_file),
            "cyclomatic_complexity": self.measure_cyclomatic_complexity(code_file),
            "test_case_coverage": "85%",  # Placeholder
            "boundary_condition_testing": "90%",  # Placeholder
            "readability": self.check_readability(code_file),
            "formatting": self.check_formatting(code_file)
        }
        return test_results

    def measure_runtime(self, code_file):
        # Measure runtime execution using line_profiler
        profiler = LineProfiler()
        exec(open(code_file).read(), globals())
        profiler.print_stats()
        return "10ms"  # Placeholder

    def measure_memory_usage(self, code_file):
        # Measure memory usage using tracemalloc
        tracemalloc.start()
        exec(open(code_file).read(), globals())
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return f"{peak / 10**6} MB"

    def measure_cpu_utilization(self, code_file):
        # Measure CPU utilization using psutil
        start_time = time.time()
        exec(open(code_file).read(), globals())
        end_time = time.time()
        cpu_percent = psutil.cpu_percent(interval=(end_time - start_time))
        return f"{cpu_percent}%"

    def measure_cyclomatic_complexity(self, code_file):
        # Placeholder for cyclomatic complexity measurement
        return "5"  # Placeholder

    def check_readability(self, code_file):
        # Check readability using pylint
        pylint_opts = ['--disable=all', '--enable=C0114,C0116,C0103']
        pylint.lint.Run([code_file] + pylint_opts, exit=False)
        return "Code is readable but can be improved."

    def check_formatting(self, code_file):
        # Check formatting using pylint
        pylint_opts = ['--disable=all', '--enable=C0330,C0301']
        pylint.lint.Run([code_file] + pylint_opts, exit=False)
        return "Code formatting needs adjustments."


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Code Reviewer Agent 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CustomCodeReviewerAgent(ConversableAgent):
    def __init__(self, name, system_message, llm_config, human_input_mode):
        super().__init__(name, system_message, llm_config, human_input_mode)
        self.best_code = ""
        self.best_test_results = None
    
    def review_test_results(self, test_results):
        # Simulate code review and provide optimization suggestions
        suggestions = {
            "readability": "Improve variable naming and add comments.",
            "formatting": "Follow PEP8 standards for formatting."
        }
        return suggestions
    
    def finalize_code(self, code, test_results):
        self.best_code = code
        self.best_test_results = test_results



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initiate the Agents:
# Reuqest Manager Agent, Planner Agent, Engineer Agent, Template Agent, Test & Modifer Agent, Code & Review Agent
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initiate the Request Manager Agent
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
request_manager_agent = CustomRequestManagerAgent(
    name="RequestManagerAgent",
    system_message="""
    You are the RequestManagerAgent. Your task is to interact with the user to understand and confirm their requirements for financial algorithms. 
    Follow these steps:
    1. Greet the user and ask them to specify the financial algorithm they need, including strategy, symbol, action, size, and maxtime.
    2. Confirm the details of their request.
    3. Provide a summary of their request and ask for confirmation.
    4. Once confirmed, prepare the request to be processed by the PlannerAgent.
    Remember to be polite, clear, and concise in your interactions.
    """,
    llm_config=llm_config,
    human_input_mode="ALWAYS",
)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initiate the Planner Agent
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
planner_agent = CustomPlannerAgent(
    name="PlannerAgent",
    system_message="""
    You are the PlannerAgent. Your task is to receive the financial algorithm requests from the RequestManagerAgent and optimize the strategies for these algorithms. 
    Follow these steps:
    1. Receive the financial algorithm request, including strategy, symbol, action, size, and maxtime.
    2. Analyze the request and provide optimization suggestions.
    3. Discuss back and forth with the RequestManagerAgent until the best possible algorithm is determined.
    4. Store the best possible algorithm in a variable and terminate the discussion.
    5. Pass the best possible algorithm to the EngineerAgent for code generation.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initiate the engineer agent
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
engineer_agent = CustomEngineerAgent(
    name="EngineerAgent",
    system_message="""
    You are the EngineerAgent. Your task is to generate Python code based on the best possible algorithm provided by the PlannerAgent.
    
    If you receive suggestions from the TemplateAgent, please modify the code based on the suggestions and resubmit it for review.
    
    If you receive compilation error messages, please modify the code based on the error messages and resubmit it for compilation.
    
    Continue this process until the code meets the template requirements and compiles successfully.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initiate the Template agent
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
template_agent = CustomTemplateAgent(
    name="TemplateAgent",
    system_message="""
    You are the TemplateAgent. Your task is to review the code generated by the EngineerAgent to ensure it meets the code structure and style requirements of the template.
    
    If the code does not meet the requirements, provide suggestions to the EngineerAgent for modifications.
    
    Once the code meets the template requirements, compile the code.
    
    If the compilation fails, provide the compilation error messages to the EngineerAgent for modifications.
    
    Once the code compiles successfully, pass it to the Tester and Modifier Agent for testing.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initiate the tester and modifier agent
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
tester_modifier_agent = CustomTesterModifierAgent(
    name="TesterModifierAgent",
    system_message="""
    You are the TesterModifierAgent. Your task is to test the code received from the TemplateAgent using various metrics such as runtime execution, memory usage, CPU utilization, test case coverage, boundary condition testing, readability, and formatting.
    
    Use appropriate tools like pylint for code errors and readability, line profiler for runtime execution, and other relevant tools for the tests.
    
    Once the testing is complete, pass the results to the CodeReviewerAgent.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initiate the code and reviewer agent
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
code_reviewer_agent = CustomCodeReviewerAgent(
    name="CodeReviewerAgent",
    system_message="""
    You are the CodeReviewerAgent. Your task is to review the test results provided by the TesterModifierAgent and give optimization suggestions.
    
    Conduct a back-and-forth discussion with the TesterModifierAgent for up to 10 iterations to achieve the best possible code.
    
    Once the best code is achieved, finalize it and store the best testing results.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Run the Scripts
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    user_input = {
        "strategy": "TWAP",
        "symbol": "ZNH0:MBO",
        "action": "buy",
        "size": 1000,
        "maxtime": 120
    }
    
    # Simulate user interaction with RequestManagerAgent
    response = request_manager_agent.process_request(user_input)
    print(response)
    
    # Simulate user confirmation
    confirmation = "yes"
    print(request_manager_agent.confirm_request(confirmation))
    
    if request_manager_agent.confirmed_request:
        best_algorithm = planner_agent.optimize_strategy(request_manager_agent.confirmed_request)
        print(f"Best Algorithm: {best_algorithm}")
        
        # Generate code based on the best algorithm
        code = engineer_agent.generate_code(best_algorithm)
        print(f"Generated Code: {code}")
        
        # Pass the generated code to TemplateAgent for review and compilation
        review_result = template_agent.review_code(code)
        while review_result != "Code meets the template requirements.":
            print(f"Review Result: {review_result}")
            code = engineer_agent.generate_code(best_algorithm)  # Simulate EngineerAgent modifying code
            review_result = template_agent.review_code(code)
        
        compile_result = template_agent.compile_code(code)
        while compile_result != "Compilation successful. Code is ready for testing.":
            print(f"Compile Result: {compile_result}")
            code = engineer_agent.generate_code(best_algorithm)  # Simulate EngineerAgent modifying code
            compile_result = template_agent.compile_code(code)
        
        # Save the final code to a .py file
        code_file = template_agent.save_code(code)
        print(f"Final Code saved to: {code_file}")
        
        # Pass the .py file to Tester & Modifier Agent for testing
        test_results = tester_modifier_agent.test_code(code_file)
        print(f"Test Results: {test_results}")
