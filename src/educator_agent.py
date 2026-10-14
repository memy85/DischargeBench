
import jinja2

from virtual_patient import vllm_response, get_answer, validate_prompt_keys


class EducatorAgent:

    def __init__(self,
                 backend_str,
                 backend_port,
                 discharge_note: dict,
                 prompt_path: str,
                 client_params=None):
        self.backend_str = backend_str
        self.backend_port = backend_port
        self.discharge_note = discharge_note
        self.prompt_path = prompt_path
        self.client_params = client_params if client_params is not None else {}

        with open(self.prompt_path, "r") as f:
            template_str = f.read()
        self.template = jinja2.Template(template_str)

        self.client = vllm_response
        self.reset()

    def set_system_prompt(self) -> None:
        self.system_prompt = self.template.render(**self.discharge_note)
        validate_prompt_keys(self.template, self.discharge_note)

    def reset(self) -> None:
        self.set_system_prompt()
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def generate_response(self, patient_response: str) -> str:
        self.messages.append({"role": "user", "content": patient_response})
        response = self.client(message=self.messages, model=self.backend_str, port=self.backend_port, **self.client_params)
        answer = get_answer(response)
        self.messages.append({"role": "assistant", "content": answer})
        return answer

