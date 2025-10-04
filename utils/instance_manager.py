from pathlib import Path
from typing import Optional, Literal, Iterator
from utils.pdptw_problem import PDPTWProblem
from utils.li_lim_reader import li_lim_reader

class InstanceManager:
    """
    Manager für Li & Lim PDPTW Benchmark-Instanzen.
    Ermöglicht Navigation und Iteration durch Instanzen verschiedener Größen und Kategorien.
    """
    
    # Verfügbare Problemgrößen
    SIZES = [100, 200, 400, 600, 1000]
    
    # Instanz-Namen pro Kategorie
    CATEGORIES = {
        'lc1': ['lc101', 'lc102', 'lc103', 'lc104', 'lc105', 'lc106', 'lc107', 'lc108', 'lc109'],
        'lc2': ['lc201', 'lc202', 'lc203', 'lc204', 'lc205', 'lc206', 'lc207', 'lc208'],
        'lr1': ['lr101', 'lr102', 'lr103', 'lr104', 'lr105', 'lr106', 'lr107', 'lr108', 'lr109', 'lr110', 'lr111', 'lr112'],
        'lr2': ['lr201', 'lr202', 'lr203', 'lr204', 'lr205', 'lr206', 'lr207', 'lr208', 'lr209', 'lr210', 'lr211'],
        'lrc1': ['lrc101', 'lrc102', 'lrc103', 'lrc104', 'lrc105', 'lrc106', 'lrc107', 'lrc108'],
        'lrc2': ['lrc201', 'lrc202', 'lrc203', 'lrc204', 'lrc205', 'lrc206', 'lrc207', 'lrc208']
    }
    
    def __init__(self, base_dir: str = 'data'):
        """
        Args:
            base_dir: Basis-Verzeichnis mit pdp_100, pdp_200, etc. Unterordnern
        """
        self.base_dir = Path(base_dir)
        self.current_size = 100
        self.current_category = 'lc1'
        self.current_index = 0
        self._problem_cache = {}
    
    def _get_categories(self, size: int) -> dict:
        if size == 100:
            # Original-Namen für size 100
            return {
                'lc1': ['lc101', 'lc102', 'lc103', 'lc104', 'lc105', 'lc106', 'lc107', 'lc108', 'lc109'],
                'lc2': ['lc201', 'lc202', 'lc203', 'lc204', 'lc205', 'lc206', 'lc207', 'lc208'],
                'lr1': ['lr101', 'lr102', 'lr103', 'lr104', 'lr105', 'lr106', 'lr107', 'lr108', 'lr109', 'lr110', 'lr111', 'lr112'],
                'lr2': ['lr201', 'lr202', 'lr203', 'lr204', 'lr205', 'lr206', 'lr207', 'lr208', 'lr209', 'lr210', 'lr211'],
                'lrc1': ['lrc101', 'lrc102', 'lrc103', 'lrc104', 'lrc105', 'lrc106', 'lrc107', 'lrc108'],
                'lrc2': ['lrc201', 'lrc202', 'lrc203', 'lrc204', 'lrc205', 'lrc206', 'lrc207', 'lrc208']
            }
        else:
            # Für 200, 400, 600, 1000: Format LR1_SIZE_NUM
            size_code = size // 100
            return {
                'lc1': [f'LC1_{size_code}_{i}' for i in range(1, 11)],
                'lc2': [f'LC2_{size_code}_{i}' for i in range(1, 11)],
                'lr1': [f'LR1_{size_code}_{i}' for i in range(1, 11)],
                'lr2': [f'LR2_{size_code}_{i}' for i in range(1, 11)],
                'lrc1': [f'LRC1_{size_code}_{i}' for i in range(1, 11)],
                'lrc2': [f'LRC2_{size_code}_{i}' for i in range(1, 11)]
            }

    def _get_path(self, instance_name: str, size: int) -> Path:
        """Pfad zur Datei - Namen sind schon korrekt."""
        return self.base_dir / f'pdp_{size}' / f'{instance_name}.txt'
    
    def load(self, instance_name: str, size: Optional[int] = None) -> PDPTWProblem:
        """
        Lädt eine spezifische Instanz.
        
        Args:
            instance_name: Name der Instanz (z.B. 'lc101')
            size: Problemgröße (default: current_size)
        """
        if size is None:
            size = self.current_size
        
        cache_key = (instance_name, size)
        if cache_key not in self._problem_cache:
            path = self._get_path(instance_name, size)
            if not path.exists():
                raise FileNotFoundError(f"Instance file not found: {path}")
            self._problem_cache[cache_key] = li_lim_reader(str(path))
        
        return self._problem_cache[cache_key]
    
    def current(self) -> PDPTWProblem:
        """Gibt das aktuelle Problem zurück."""
        instance_name = self.CATEGORIES[self.current_category][self.current_index]
        return self.load(instance_name, self.current_size)
    
    def next(self) -> PDPTWProblem:
        """
        Springt zur nächsten Instanz in der aktuellen Kategorie und Größe.
        Wechselt automatisch zur nächsten Kategorie wenn am Ende.
        """
        category_instances = self.CATEGORIES[self.current_category]
        self.current_index += 1
        
        if self.current_index >= len(category_instances):
            # Zur nächsten Kategorie
            categories = list(self.CATEGORIES.keys())
            current_cat_idx = categories.index(self.current_category)
            
            if current_cat_idx < len(categories) - 1:
                self.current_category = categories[current_cat_idx + 1]
                self.current_index = 0
            else:
                # Zurück zum Start
                self.current_category = categories[0]
                self.current_index = 0
        
        return self.current()
    
    def prev(self) -> PDPTWProblem:
        """Springt zur vorherigen Instanz."""
        self.current_index -= 1
        
        if self.current_index < 0:
            # Zur vorherigen Kategorie
            categories = list(self.CATEGORIES.keys())
            current_cat_idx = categories.index(self.current_category)
            
            if current_cat_idx > 0:
                self.current_category = categories[current_cat_idx - 1]
                self.current_index = len(self.CATEGORIES[self.current_category]) - 1
            else:
                # Zum Ende springen
                self.current_category = categories[-1]
                self.current_index = len(self.CATEGORIES[self.current_category]) - 1
        
        return self.current()
    
    def jump_to_size(self, size: int) -> 'InstanceManager':
        """
        Wechselt zur angegebenen Problemgröße.
        
        Args:
            size: Eine der verfügbaren Größen (100, 200, 400, 600, 1000)
        """
        if size not in self.SIZES:
            raise ValueError(f"Size {size} not available. Choose from {self.SIZES}")
        self.current_size = size
        return self
    
    def jump_to_category(self, category: str) -> 'InstanceManager':
        """
        Wechselt zur angegebenen Kategorie.
        
        Args:
            category: Eine der Kategorien ('lc1', 'lc2', 'lr1', 'lr2', 'lrc1', 'lrc2')
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"Category {category} not available. Choose from {list(self.CATEGORIES.keys())}")
        self.current_category = category
        self.current_index = 0
        return self
    
    def jump_to(self, instance_name: str) -> 'InstanceManager':
        """
        Springt direkt zu einer benannten Instanz.
        
        Args:
            instance_name: Name der Instanz (z.B. 'lc201')
        """
        for category, instances in self.CATEGORIES.items():
            if instance_name in instances:
                self.current_category = category
                self.current_index = instances.index(instance_name)
                return self
        
        raise ValueError(f"Instance {instance_name} not found")
    
    def get_all_in_category(self, category: str, size: Optional[int] = None) -> list[PDPTWProblem]:
        """
        Lädt alle Instanzen einer Kategorie.
        
        Args:
            category: Kategorie-Name
            size: Problemgröße (default: current_size)
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        
        if size is None:
            size = self.current_size
            
        problems = []
        for instance_name in self.CATEGORIES[category]:
            problems.append(self.load(instance_name, size))
        
        return problems
    
    def get_all(self, size: Optional[int] = None) -> list[PDPTWProblem]:
        """
        Lädt alle verfügbaren Instanzen einer Größe.
        
        Args:
            size: Problemgröße (default: current_size)
        """
        if size is None:
            size = self.current_size
            
        problems = []
        for category in self.CATEGORIES.keys():
            problems.extend(self.get_all_in_category(category, size))
        
        return problems
    
    def iterate_current(self) -> Iterator[tuple[str, int, PDPTWProblem]]:
        """
        Iteriert nur über Instanzen der aktuellen Größe und Kategorie.
        
        Yields:
            (instance_name, size, problem) tuples
        """
        for instance_name in self.CATEGORIES[self.current_category]:
            problem = self.load(instance_name, self.current_size)
            yield (instance_name, self.current_size, problem)
    
    def iterate_category(self, category: str, size: Optional[int] = None) -> Iterator[tuple[str, int, PDPTWProblem]]:
        """
        Iteriert über alle Instanzen einer Kategorie.
        
        Args:
            category: Kategorie-Name
            size: Problemgröße (default: current_size)
            
        Yields:
            (instance_name, size, problem) tuples
        """
        if size is None:
            size = self.current_size
            
        for instance_name in self.CATEGORIES[category]:
            problem = self.load(instance_name, size)
            yield (instance_name, size, problem)
    
    def iterate_size(self, size: int) -> Iterator[tuple[str, str, int, PDPTWProblem]]:
        """
        Iteriert über alle Instanzen ALLER Kategorien einer bestimmten Größe.
        
        Args:
            size: Problemgröße
            
        Yields:
            (instance_name, category, size, problem) tuples
        """
        for category, instances in self.CATEGORIES.items():
            for instance_name in instances:
                problem = self.load(instance_name, size)
                yield (instance_name, category, size, problem)
    
    def iterate_all(self, 
                   sizes: Optional[list[int]] = None,
                   categories: Optional[list[str]] = None) -> Iterator[tuple[str, str, int, PDPTWProblem]]:
        """
        Iteriert über ALLE Instanzen aller Größen und Kategorien.
        
        Args:
            sizes: Liste von Größen (default: alle verfügbaren)
            categories: Liste von Kategorien (default: alle verfügbaren)
            
        Yields:
            (instance_name, category, size, problem) tuples
        """
        if sizes is None:
            sizes = self.SIZES
        if categories is None:
            categories = list(self.CATEGORIES.keys())
        
        for size in sizes:
            for category in categories:
                if category not in self.CATEGORIES:
                    continue
                for instance_name in self.CATEGORIES[category]:
                    problem = self.load(instance_name, size)
                    yield (instance_name, category, size, problem)
    
    def __repr__(self) -> str:
        instance_name = self.CATEGORIES[self.current_category][self.current_index]
        return f"InstanceManager(current={instance_name}, size={self.current_size}, category={self.current_category})"

