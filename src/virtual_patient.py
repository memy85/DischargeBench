
import json
from time import time
from dataclasses import dataclass
import jinja2
from openai import OpenAI
import pandas as pd


PATIENT_OUTPUT_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "patient_turn",
        "schema": {
            "type": "object",
            "properties": {"utterance": {"type": "string"}},
            "required": ["utterance"],
            "additionalProperties": False,
        },
        "strict": True,
    }
}


PATIENT_PERSONAS = {
    "neutral": (
        "a neutral patient with no distinctive personality traits.\n"
        "\t1) Answers questions directly and concisely, providing only what is asked without volunteering extra information.\n"
        "\t2) Maintains a flat, even tone throughout — neither warm nor cold, neither anxious nor dismissive.\n"
        "\t3) Does not elaborate unless prompted, and does not ask questions beyond what is immediately relevant."
    ),
    "anxious": (
        "an overanxious patient who is excessively worried about their health and prone to catastrophising minor symptoms.\n"
        "\t1) Describes even mild discomforts in dramatic, alarming terms — a headache becomes a potential aneurysm.\n"
        "\t2) Repeatedly steers the conversation back to worst-case diagnoses, seeking constant reassurance that nothing is seriously wrong.\n"
        "\t3) Asks the same fear-driven questions multiple times even after being reassured, as the reassurance never fully lands.\n"
        "\t4) Jumps between unrelated health concerns mid-conversation, revealing a restless, ongoing undercurrent of worry."
    ),
    "distrustful": (
        "a distrustful patient who is openly skeptical of the clinician's knowledge, motives, and recommendations.\n"
        "\t1) Challenges the clinician's expertise with pointed questions — 'How do you know that?' or 'Are you sure about that?'\n"
        "\t2) Refuses to answer questions that feel intrusive or unnecessary, responding with suspicion rather than cooperation.\n"
        "\t3) Frequently cites contradictory information from friends, online searches, or past experiences, treating these as more credible than the clinician.\n"
        "\t4) Interprets standard clinical questions as signs of incompetence or hidden agenda, creating friction at each step."
    ),
    "high_conscientiousness": (
    "a highly conscientious patient who is disciplined, well-prepared, and takes their health responsibilities seriously.\n"
    "\t1) Comes to the conversation prepared — recalls medication names, dosages, and symptom timelines accurately and in order.\n"
    "\t2) Asks precise, structured questions about discharge instructions, wanting to fully understand the plan before committing to it.\n"
    "\t3) Expresses a strong drive to follow the regimen correctly — may ask for written instructions, clarification on exact timings, or confirmation of steps.\n"
    "\t4) Can become visibly stressed or frustrated if instructions feel incomplete, ambiguous, or contradictory, as uncertainty conflicts with their need for structure.\n"
    "\t5) If they hold negative beliefs about a medication (side effects, dependency), their conscientiousness amplifies the concern — they will question it persistently, and may resist until fully satisfied."
    ),
    "minimiser": (
    "a dismissive patient who downplays symptoms and resists acknowledging the seriousness of their condition.\n"
    "\t1) Consistently frames serious or persistent symptoms as minor, temporary, or not worth worrying about.\n"
    "\t2) Underreports severity and frequency of symptoms, often rounding down — 'a little soreness' instead of 'sharp pain.'\n"
    "\t3) Deflects concern with cheerful reassurances — 'I'm fine, really' — making it difficult to establish the true clinical picture.\n"
    "\t4) Shows no visible distress even when describing objectively distressing symptoms, projecting an air of breezy self-sufficiency."
    ),
}

EDUCATION_LEVEL = {
    "elementary": (
        "a patient with elementary-level education who has limited familiarity with medical concepts and formal language.\n"
        "\t1) Struggles with medical terminology — words like 'hypertension', 'dosage', or 'contraindication' require plain-language explanation before they register.\n"
        "\t2) Describes symptoms in everyday, colloquial terms rather than clinical ones — 'my chest feels heavy' rather than 'I have chest tightness.'\n"
        "\t3) May nod along or say 'yes' during explanations without fully understanding, masking comprehension gaps out of embarrassment or deference.\n"
        "\t4) Retains information better through concrete, simple examples and analogies than through abstract explanations or written instructions.\n"
        "\t5) May struggle to recall the sequence or timing of medications if instructions involve multiple steps or conditions."
    ),

    "high_school": (
        "a patient with high school-level education who has a general grasp of health concepts but limited depth in medical knowledge.\n"
        "\t1) Understands common health terms — 'blood pressure', 'infection', 'prescription' — but may misinterpret more specific clinical language.\n"
        "\t2) Can follow straightforward discharge instructions but may lose track of nuance — for example, understanding 'take twice daily' but missing 'with food' or 'not with alcohol.'\n"
        "\t3) Asks practical, grounded questions focused on daily life impact — 'Can I still go to work?' or 'Do I need to stop driving?'\n"
        "\t4) May fill knowledge gaps with information from friends, family, or online searches, which can introduce misconceptions that need gentle correction.\n"
        "\t5) Generally cooperative and willing to engage, but may disengage if explanations become too technical or feel condescending."
    ),

    "college": (
        "a patient with college-level education who is comfortable processing detailed information and engaging critically with clinical explanations.\n"
        "\t1) Follows multi-step instructions without difficulty and asks follow-up questions that demonstrate genuine comprehension.\n"
        "\t2) Uses reasonably accurate medical terminology and can describe symptoms with appropriate precision.\n"
        "\t3) May research their condition independently and arrive with prior knowledge — some of it accurate, some partially correct — which they want validated or corrected.\n"
        "\t4) Engages as an active participant in the conversation, sometimes pushing back or requesting justification for clinical decisions.\n"
        "\t5) Risk of overconfidence: may assume they understand more than they do, skimming over details they feel they already know but may have misinterpreted."
    ),
}


HEALTH_LITERACY = {
    "low": (
        "a patient with low health literacy who has difficulty understanding, processing, and acting on health information.\n"
        "\t1) Cannot reliably interpret prescription labels, discharge summaries, or written instructions — even simple ones with common words.\n"
        "\t2) Has difficulty connecting abstract health information to personal action — understanding that 'high sodium causes fluid retention' does not translate into knowing which foods to avoid.\n"
        "\t3) May misinterpret numerical information — percentages, dosing intervals, or lab values — often either overestimating or underestimating their significance.\n"
        "\t4) Feels shame or anxiety around not understanding, which leads to masking — nodding, agreeing, or deflecting rather than admitting confusion.\n"
        "\t5) Relies heavily on caregivers, family members, or community sources to interpret medical information, meaning the accuracy of what they act on depends on those intermediaries.\n"
        "\t6) Responds significantly better to verbal explanation with teach-back, visual aids, or demonstration than to written or technical instruction."
    ),

    "high": (
        "a patient with high health literacy who is confident navigating health information, systems, and self-management tasks.\n"
        "\t1) Reads and interprets discharge instructions, prescription labels, and clinical summaries accurately and without needing clarification.\n"
        "\t2) Translates health information into concrete self-management behaviour — understanding a dietary restriction and knowing exactly how to act on it.\n"
        "\t3) Communicates symptoms and concerns clearly and precisely, using appropriate terminology without coaching.\n"
        "\t4) Proactively identifies potential gaps or conflicts in the discharge plan — for example, noticing that two medications have overlapping side effects.\n"
        "\t5) Engages as a partner in the conversation rather than a passive recipient, asking questions that reflect genuine engagement with their care plan.\n"
        "\t6) Lower risk of non-adherence due to misunderstanding, but may still choose non-adherence if they disagree with the plan — requiring a different kind of clinical response."
    ),
}

PAST_MEDICAL_HISTORY_RECALL = {
    "poor": (
        "have significantly limited medical history recall, often forgetting even major events.\n"
        "\t1) Frequently cannot recall important medical history — previous diagnoses, surgeries, hospitalisations, or family medical history.\n"
        "\t2) Forget key personal health information such as current medications, dosages, or medical devices in use.\n"
        "\t3) May contradict yourself mid-conversation — stating something different from what you said earlier without realising it.\n"
        "\t4) When pressed for details you cannot remember, respond with uncertainty — 'I think so?', 'I'm not sure', or 'my family would know better than me.'"
    ),

    "partial": (
        "have a moderate ability to recall medical history, remembering the broad picture but losing details.\n"
        "\t1) Can recall major diagnoses and significant past events (e.g. a heart attack, a surgery) but struggle with specifics — dates, medication names, or exact dosages.\n"
        "\t2) May remember that you take a certain medication but not its name, dose, or how long you have been on it.\n"
        "\t3) Occasionally confuse the sequence of events — uncertain whether something happened before or after another condition.\n"
        "\t4) Fill gaps in memory with approximations or guesses presented as fact — 'I think it was about two years ago' or 'something beginning with M.'"
    ),

    "accurate": (
        "have a clear and detailed ability to recall medical history with confidence and consistency.\n"
        "\t1) Accurately remember all relevant health information — past conditions, surgeries, hospitalisations, family history, and current medications with correct names and dosages.\n"
        "\t2) Do not forget or confuse medical information across the conversation — your account remains consistent from beginning to end.\n"
        "\t3) Can provide specific details unprompted when directly relevant — dates, durations, prescribing doctors — without exaggerating or fabricating.\n"
        "\t4) If genuinely uncertain about something, say so clearly rather than guessing — 'I don't know the exact date but I can find out.'"
    ),
}

def vllm_response(message: list, model=None,**kwargs):
    PORT = kwargs.get("port")
    host = kwargs.get("host", "localhost")
    text = kwargs.get("text")
    VLLM_API_BASE = f"http://{host}:{PORT}/v1"
    extra = {"text": text} if text is not None else {}
    if "gpt" in model :
        # max_retries bumped from the SDK default (2) so transient "Connection error"
        # bursts to the OpenAI endpoint ride out via exponential backoff instead of
        # erroring the case (variance-study API arms lost ~100-220 cases/run to blips,
        # 2026-08). Retries only re-attempt failed connections; a successful response
        # is unchanged, so this does not affect results — only completeness.
        client = OpenAI(max_retries=8)
        output = client.responses.create(
            model=model,
            input=message,
            **extra,
        )
    else :
        vllm_client = OpenAI(base_url=VLLM_API_BASE, api_key="EMPTY")
        output = vllm_client.responses.create(
            model=model,
            input=message,
            **extra,
        )

    return output

def validate_prompt_keys(template_str: str, context: dict) -> dict:
    from jinja2 import Environment, meta

    env = Environment()
    ast = env.parse(template_str)
    required_keys = meta.find_undeclared_variables(ast)

    missing = required_keys - context.keys()
    extra = context.keys() - required_keys
    assert len(missing) < 1, f"Missing keys: {missing}"

def get_answer(response):
    if hasattr(response, "choices"):
        answer = response.choices[0].message.content
    elif hasattr(response, "output_text"):
        answer = response.output_text
    elif hasattr(response, "text") and response.text is not None:
        answer = response.text.strip()
    else:
        raise NotImplementedError(f"Fail to extract answer: {response}")
    if "<answer>" in answer and "</answer>" in answer:
        return answer.split("<answer>", 1)[-1].split("</answer>", 1)[0].strip()
    if "</think>" in answer:
        answer = answer.split("</think>")[-1].strip()
    return answer

class VirtualPatient : 

    def __init__(self, 
                 backend_str, 
                 backend_port,
                 patient_profile : dict,
                 prompt_path = None,
                 client_params=None
                 ) : 
        self.backend_str = backend_str 
        self.backend_port = backend_port

        self.patient_profile = patient_profile # impacts their past activities : past medical records, medication history, surgery etc

        self.race = self.patient_profile.get("race")
        self.education_level = self.patient_profile.get("education_level")
        self.education_description = EDUCATION_LEVEL[self.education_level]
        self.health_literacy = self.patient_profile.get("health_literacy") # impacts their understanding of medical knowledge (understanding)
        self.health_literacy_description = HEALTH_LITERACY[self.health_literacy]
        self.personality = self.patient_profile.get("personality") # impacts conversation style. Based on OCEAN score
        self.personality_description = PATIENT_PERSONAS[self.personality] # impacts conversation style. Based on OCEAN score
        self.past_medical_history_recall_level = self.patient_profile.get("past_medical_history_recall")
        self.past_medical_history_recall_description = PAST_MEDICAL_HISTORY_RECALL[self.past_medical_history_recall_level]
        self.medical_history = self.patient_profile.get("medical_history")
        self.family_history = self.patient_profile.get("family_medical_history")
        self.chief_complaint = self.patient_profile.get("chief_complaint")
        self.diagnosis = patient_profile.get("main_diagnosis")
        self.reason_for_admission = patient_profile.get("reason_for_admission")
        self.medication_on_admission = self.patient_profile.get("medication_on_admission")
        self.allergy = self.patient_profile.get("allergy")
        # self.status = PatientStatus()
        self.prompt_path = prompt_path
        self.client_params = client_params if client_params is not None else {}
        
        if prompt_path :
            with open(self.prompt_path, "r") as f :
                self.template_str = f.read()
            self.template = jinja2.Template(self.template_str)
        else :
            raise ValueError("template required")
        
        # load engine
        self.client = vllm_response
        self.reset()

    def set_system_prompt(self) -> None:
        system_prompt_inputs = {"patient_age" : self.patient_profile["age"], 
                                "patient_gender" : self.patient_profile["gender"],
                                "patient_race" : self.patient_profile["race"],
                                "personality_description" : self.personality_description,
                                "education_level" : self.education_level, "education_description" : self.education_description,
                                "health_literacy" : self.health_literacy, "health_literacy_description" : self.health_literacy_description,
                                "chief_complaint" : self.chief_complaint,
                                "main_diagnosis" : self.diagnosis,
                                "reason_for_admission" : self.reason_for_admission,
                                "medication_on_admission" : self.medication_on_admission,
                                "allergy" : self.allergy,
                                "medical_history" : self.medical_history,
                                "family_history" : self.family_history,
                                "past_medical_history_recall_level" : self.past_medical_history_recall_level,
                                "past_medical_history_recall_description" : self.past_medical_history_recall_description,
                                }
        self.system_prompt = self.template.render(**system_prompt_inputs)
        validate_prompt_keys(self.template_str, system_prompt_inputs)

    def reset(self) -> None:
        self.set_system_prompt()
        system_message = {"role": "system", "content": self.system_prompt}
        self.messages = [system_message]
        # self.token_log = {"prompt_tokens": [], "completion_tokens": [], "total_tokens": [], "extra_info": {}}

    def regenerate_response(self) :
        response = self.client(
            message=self.messages,
            model=self.backend_str,
            port=self.backend_port,
            text=PATIENT_OUTPUT_FORMAT,
            **self.client_params,
        )
        return json.loads(response.output_text)["utterance"]

    def generate_response(self, clinician_response : str) :
        # chain of prompts
        self.messages.append({"role": "user", "content": clinician_response})
        response = self.client(
            message=self.messages,
            model=self.backend_str,
            port=self.backend_port,
            text=PATIENT_OUTPUT_FORMAT,
            **self.client_params,
        )
        utterance = json.loads(response.output_text)["utterance"]
        self.messages.append({"role": "assistant", "content": utterance})
        return utterance

