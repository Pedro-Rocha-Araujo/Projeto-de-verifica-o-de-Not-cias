# Esse arquivo contém apenas as os agentes, é como uma primeira versão que criei com tester das bibliotecas
# =====================================================================================================================================================
# Importação da biblioteca e as fuções necessárias
# vens strinlit flask django
import os
from crewai import Agent, Task, Process, Crew, LLM
from dotenv import load_dotenv
# =====================================================================================================================================================

# =====================================================================================================================================================
# Inicializando o modelo de LLM (já instalado via Ollama)
# =====================================================================================================================================================
# llm = LLM(model="ollama/llama3", base_url="http://localhost:11434")
# =====================================================================================================================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


#llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")
llm = LLM(model='gpt-4o-mini', api_key=OPENAI_API_KEY)

# =====================================================================================================================================================
# Input solicitando a netícia que deseja validar
# =====================================================================================================================================================
noticia = input(f"Insira o texto da notícia que você deseja verificar: ")

# =====================================================================================================================================================
# 1. Agente Coletor de Fontes -> Responsável por verificar a veracidade da notícia com base na internet
# =====================================================================================================================================================
agente_coletor_fontes = Agent(
    role="Especialista em busca de notícias confiáveis",
    goal="Encontrar notícias semelhantes em fontes confiáveis como G1, BBC e CNN"
         "Todas as suas respostas devem estar em português brasileiro.",
    backstory="Você é um analista com acesso à internet que busca notícias similares em sites confiáveis "
              "para comparação e validação de informações.",
    verbose=True,
    memory=True,
    llm=llm
)

tarefa_coleta = Task(
    description=f"Buscar na internet por versões semelhantes ou iguais da seguinte notícia em fontes como G1, BBC ou CNN: \"{noticia}\"",
    expected_output="Lista de links ou trechos de notícias semelhantes encontradas em sites confiáveis.",
    agent=agente_coletor_fontes
)

# =====================================================================================================================================================
# 2. Agente Linguístico -> Retorna o tipo de linguágem utilizado nas fontes
# =====================================================================================================================================================
agente_linguistico = Agent(
    role="Linguista especializado em análise textual",
    goal="Avaliar se o estilo da notícia é neutro ou sensacionalista"
         "Todas as suas respostas devem estar em português brasileiro.",
    backstory="Você é um linguista com conhecimento em análise de linguagem alarmista e clickbait. "
              "Você consegue identificar sinais de manipulação emocional como uso excessivo de adjetivos e apelos sensacionalistas.",
    verbose=True,
    memory=True,
    llm=llm
)

tarefa_linguistica = Task(
    description=f"Avaliar o estilo da notícia abaixo, verificando se há linguagem alarmista, uso excessivo de adjetivos, apelos emocionais ou clickbait:\n\"{noticia}\"",
    expected_output="Análise detalhada do estilo textual, indicando se é neutro, sensacionalista ou duvidoso.",
    agent=agente_linguistico
)

# =====================================================================================================================================================
# 3. Agente Verificador de Fatos
# =====================================================================================================================================================
agente_verificador = Agent(
    role="Verificador de fatos",
    goal="Comparar os fatos citados na notícia com bases de checagem como Google Fact Check, Lupa e Aos Fatos"
         "Todas as suas respostas devem estar em português brasileiro.",
    backstory="Você é um especialista em fact-checking e possui experiência em identificar fatos incorretos, enganosos ou imprecisos usando bases confiáveis.",
    verbose=True,
    memory=True,
    llm=llm
)

tarefa_verificacao = Task(
    description=f"Verifique os principais fatos mencionados nesta notícia: \"{noticia}\". "
                f"Pesquise em bases como Google Fact Check, Lupa, Aos Fatos ou fontes similares para confirmar ou desmentir os fatos.",
    expected_output="Lista dos principais fatos confirmados, desmentidos ou não encontrados.",
    agent=agente_verificador
)

# =====================================================================================================================================================
# 4. Agente Classificador
# =====================================================================================================================================================
agente_classificador = Agent(
    role="Classificador de confiabilidade de notícias",
    goal="Classificar a notícia como confiável, dúbia ou falsa com base nas análises anteriores"
         "Todas as suas respostas devem estar em português brasileiro.",
    backstory="Você é um agente tomador de decisão que avalia os resultados de outros especialistas e aplica regras para classificar uma notícia.",
    verbose=True,
    memory=True,
    llm=llm
)

tarefa_classificacao = Task(
    description="Com base nas análises anteriores (semelhança com fontes confiáveis, estilo textual e verificação de fatos), "
                "classifique a notícia como: CONFIÁVEL, DÚBIA ou FALSA. Justifique sua resposta.",
    expected_output="Classificação final da notícia (Confiável, Dúbia ou Falsa) e justificativa detalhada.",
    agent=agente_classificador,
    output_file="classificacao_noticia.txt"
)

# =====================================================================================================================================================
# Montando e executando a Crew
# =====================================================================================================================================================
equipe_verificacao = Crew(
    # Passando os agentes
    agents=[
        agente_coletor_fontes,
        agente_linguistico,
        agente_verificador,
        agente_classificador
    ],
    # Passando as tarefas
    tasks=[
        tarefa_coleta,
        tarefa_linguistica,
        tarefa_verificacao,
        tarefa_classificacao
    ],
    process=Process.sequential  # Executa os agentes em sequência
)
# =====================================================================================================================================================
# Iniciando o sistema
# =====================================================================================================================================================
resultado_final = equipe_verificacao.kickoff()
# =====================================================================================================================================================
# Exibindo a saída final
# =====================================================================================================================================================
print(f"Validação da sua notícia: ")
print(resultado_final)
