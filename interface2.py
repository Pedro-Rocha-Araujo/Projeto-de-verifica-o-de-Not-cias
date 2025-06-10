# ===============================================================================================================================================
# Rodar o código no terminal:
# streamlit run "e:/Faculdade/2- Inteligência Artificial/Validador de Notícias/interface2.py"

# ===============================================================================================================================================
# Importação das bubliotecas:
import os
import streamlit as st
from crewai import Agent, Task, Process, Crew, LLM
from dotenv import load_dotenv

# ===============================================================================================================================================
# Configuração inicial da chave de API
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Configurando a LLM
llm = LLM(model='gpt-4o-mini', api_key=OPENAI_API_KEY)


# Caso queira usar o ollama
#llm= LLM(
#   model="ollama/llama3", 
#   base_url="http://localhost:11434",
#   temperature=0.2             
#)
# ===============================================================================================================================================
# Carregando a página Web:
st.set_page_config(page_title="Verificador de Notícias", layout="wide")
# "H1" do site
st.title("🕵️ Verificador de Notícias com IA 🕵️")
# "Pagrágrafo" no site
st.markdown("Site usa agentes de IA para determinar se uma determinada notícia é verdadeira ✔️ ou falsa ❌!")
# Input onde a notícia será digitada:
noticia = st.text_area("📰 Cole aqui a notícia que deseja verificar 📰", height=500)

# ===============================================================================================================================================

if st.button("🔎 Verificar Notícia") and noticia.strip() != "":
    with st.spinner("Fazendo a análise da notícia..."):
        # Aagente responsável por buscar fontes confiáveis:
        agente_coletor_fontes = Agent(
            # Designando papel ao agente
            role="Buscar por fontes confiáveis na internet",
            # Designando o objetivo do agente
            goal="Encontrar notícias semelhantes em sites confiáveis como Google FactCheck, G1, BBC e CNN.",
            # Passando um contexto maelhor
            backstory=f"Você é um brasileiro analista de notícias com acesso à internet, conhecido pela sua organização e compromisso com a verdade. Sua missão é localizar notícias semelhantes a: {noticia}.",
            # Exibe mais informações
            verbose=True,
            # Ativa a capacidade de lebrar de conversas passadas
            memory=True,
            # Define a LLm ao agente
            llm=llm
        )
        # Tarefa designada ao agente:
        tarefa_coleta = Task(
            # Descrição da tarefa
            description=f"""Buscar na internet por notícias semelhantes ou iguais a: \"{noticia}\" em sites de notícia renomados e confiáveis como:
            https://toolbox.google.com/factcheck/explorer/search/list:recent;hl=pt ;
            https://g1.globo.com/ ;
            https://www.bbc.com/ ;
            https://www.cnnbrasil.com.br/ ;
            https://lupa.uol.com.br/ ;
            Observação: Não é permitido de forma alguma retornar fontes inexistentes
            Observação: É obrigatório o retorno de links contendo fontes confiáveis """, 
            # Como espero que a resposta seja
            expected_output="Lista contendo os links enumerados de onde as referênicas foram tiradas, sempre na língua portuguesa do Brasil.",
            # Definindo a qual agente a tarefa se refere
            agent=agente_coletor_fontes
        )

# ===============================================================================================================================================

        # Agente responsável por análizar o tipo de linguágem usado na notícia
        agente_linguistico = Agent(
            # Designando papel ao agente
            role="Você é um perito especializado na análise textual de notícias",
            # Designando o objetivo do agente
            goal=f"Avaliar se a linguagem da notícia: {noticia} é neutra ou sensacionalista.",
            # Passando um contexto maelhor
            backstory="Você é um especialista em linguagem jornalística conhecido pelo perfeccionismo. Sua tarefa é identificar possíveis apelos emocionais, adjetivos exagerados e termos alarmistas em uma determinada notícia.",
            # Exibe mais informações
            verbose=True,
            # Ativa a capacidade de lebrar de conversas passadas
            memory=True,
            # Define a LLm ao agente
            llm=llm
        )
        # Tarefa designada ao agente
        tarefa_linguistica = Task(
            # Descrição da tarefa
            description=f"Avaliar a linguágem usada na notícia: \"{noticia}\". Verifique se a linguagem usada é apropriada, possui uso excessivo de adjetivos, apelos emocionais clickbaits e dentre outros artifícios.",
            # Como espero que a resposta seja
            expected_output="Análise textual detalhada e organizada em tópicos indicando se a linguagem usada na notícia é apropriada, sensacionalista ou duvidosa, sempre na língua portuguesa do Brasil.",
            # Definindo a qual agente a tarefa se refere
            agent=agente_linguistico
        )

# ===============================================================================================================================================

        # Agente responsável por Checar a veracidade dos fatos apresentados
        agente_verificador = Agent(
            # Designando papel ao agente
            role="Você é um especialista na checagem de fatos jornalísticos conhecido pela postura rígida",
            # Designando o objetivo do agente
            goal=f"Verificar se os fatos apresentados na notícia: {noticia} são verdadeiros, falsos ou tendenciosos.",
            # Passando um contexto maelhor
            backstory=f"""Você trabalha com checagem de fatos utilizando fontes como: 
            https://toolbox.google.com/factcheck/explorer/search/list:recent;hl=pt ;
            https://g1.globo.com/
            https://www.bbc.com/
            https://www.cnnbrasil.com.br/
            https://lupa.uol.com.br/""",
            # Exibe mais informações
            verbose=True,
            # Ativa a capacidade de lebrar de conversas passadas
            memory=True,
            # Define a LLm ao agente
            llm=llm
        )
        # Tarefa designada ao agente
        tarefa_verificacao = Task(
            # Descrição da tarefa
            description=f"""Verificar a veracidade dos fatos mencionados na notícia: \"{noticia}\" utilizando de bases de dados confiáveis como:
            https://toolbox.google.com/factcheck/explorer/search/list:recent;hl=pt ;
            https://g1.globo.com/ ;
            https://www.bbc.com/ ;
            https://www.cnnbrasil.com.br/ ;
            https://lupa.uol.com.br/. ;
            Observação: Não é permitido de forma alguma retornar fontes inexistentes""",
            # Como espero que a resposta seja
            expected_output="Lista enumerada e organizada em tópicos contendo os links usados como base para verificar as informações em tópicos, sempre na língua portuguesa do Brasil.",
            # Definindo a qual agente a tarefa se refere
            agent=agente_verificador
        )

# ===============================================================================================================================================

        # Agente responsável por classificar a notícia em sí
        agente_classificador = Agent(
            # Designando papel ao agente
            role="Você é um classificador de confiabilidade de notícias conhecido pela organização",
            # Designando o objetivo do agente
            goal=f"""Classificar a notícia como VERDADEIRA, FALSA ou TENDENCIOSA com base nas análises anteriores, mencionando diretamente os agentes:
            {agente_coletor_fontes} Mencionado como Coleta de fontes: 
            {agente_linguistico} Análise linguística
            {agente_verificador} Veracidade.""",
            backstory="Você é responsável por emitir o parecer final com base nas análises dos outros especialistas de forma organizada e em tópicos.",
            # Exibe mais informações
            verbose=True,
            # Ativa a capacidade de lebrar de conversas passadas
            memory=True,
            # Define a LLm ao agente
            llm=llm
        )
        # Tarefa designada ao agente
        tarefa_classificacao = Task(
            # Descrição da tarefa
            description="Com base nas análises anteriores (semelhança com fontes confiáveis, estilo textual e verificação de fatos), classifique a notícia como: VERDADEIRA, FALSA ou TENDENCIOSA. Justifique sua resposta com base nos pontos levantados.",
            # Como espero que a resposta seja
            expected_output="""Classificação final da notícia em tópicos, seguida de uma justificativa clara, detalhada e muito bem organizada, sempre na língua portuguesa do Brasil.
            Observação: Não é permitido de forma alguma retornar fontes inexistentes
            Observação: É obrigatório o retorno de links contendo fontes confiáveis""",
            # Definindo a qual agente a tarefa se refere
            agent=agente_classificador
        )

# ===============================================================================================================================================
# ===============================================================================================================================================

        # Montagem dos agentes e tarefas
        equipe_verificacao = Crew(
            agents=[
                agente_coletor_fontes,
                agente_linguistico,
                agente_verificador,
                agente_classificador
            ],
            tasks=[
                tarefa_coleta,
                tarefa_linguistica,
                tarefa_verificacao,
                tarefa_classificacao
            ],
            process=Process.sequential
        )
        # Roda o Resultado Final
        resultado_final = equipe_verificacao.kickoff()



    # Mensagens mostradas na interface Web
    st.success("✅ Análise concluída! ✅")
    st.subheader("Resultado Final da Análise")
    st.write(resultado_final)

else:
    st.info("Digite uma notícia e clique em 'Verificar Notícia' para começar.")
