"""
Test cases for the serialization module, including union types that require __dataclass_type__.
"""
import unittest
import tempfile
import os
import json
import enum
from dataclasses import dataclass
from typing import Union, List, Optional, Dict
from spatialstencil.syntax.common.serialization import (DataclassEncoder, dataclass_decoder, save_to_json,
                                                        load_from_json)


# Test enums
class Color(enum.Enum):
    RED = enum.auto()
    GREEN = enum.auto()
    BLUE = enum.auto()


class Size(enum.Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


# Test dataclasses
@dataclass
class Point:
    x: int
    y: int


@dataclass
class Circle:
    center: Point
    radius: float
    color: Color


@dataclass
class Rectangle:
    top_left: Point
    width: int
    height: int
    color: Color


@dataclass
class Triangle:
    p1: Point
    p2: Point
    p3: Point
    color: Color


# Union type container that demonstrates the need for __dataclass_type__
@dataclass
class Shape:
    """Container that can hold different shape types."""
    shape: Union[Circle, Rectangle, Triangle]
    size: Size
    name: str


@dataclass
class Drawing:
    """Contains multiple shapes - this tests nested Union types."""
    shapes: List[Shape]
    title: str
    background_color: Color


@dataclass
class ComplexNested:
    """Complex nested structure with multiple union types."""
    primary_shape: Union[Circle, Rectangle]
    secondary_shapes: List[Union[Triangle, Point]]  # Mix of dataclasses
    metadata: Optional[Union[str, int]]
    config: Union[Color, Size]


@dataclass
class OptionalFields:
    """Test optional fields and defaults."""
    required_field: str
    optional_shape: Optional[Union[Circle, Rectangle]] = None
    default_color: Color = Color.RED


@dataclass
class DictContainer:
    """Test dictionary fields with union types as values."""
    shape_registry: Dict[str, Union[Circle, Rectangle, Triangle]]
    config_map: Dict[Color, Union[str, int]]
    name: str


class TestSerialization(unittest.TestCase):
    """Test cases for dataclass serialization and deserialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_simple_dataclass_encoding(self):
        """Test encoding of simple dataclass."""
        point = Point(10, 20)
        encoder = DataclassEncoder()
        result = encoder.default(point)

        expected = {'x': 10, 'y': 20, '__dataclass_type__': 'Point'}
        self.assertEqual(result, expected)

    def test_enum_encoding(self):
        """Test encoding of enum values."""
        encoder = DataclassEncoder()
        result = encoder.default(Color.RED)
        self.assertEqual(result, 'RED')

    def test_nested_dataclass_encoding(self):
        """Test encoding of nested dataclasses."""
        circle = Circle(Point(5, 10), 3.5, Color.BLUE)
        encoder = DataclassEncoder()
        result = json.dumps(circle, cls=DataclassEncoder)

        # Parse back to verify structure
        parsed = json.loads(result)
        self.assertEqual(parsed['__dataclass_type__'], 'Circle')
        self.assertEqual(parsed['center']['__dataclass_type__'], 'Point')
        self.assertEqual(parsed['center']['x'], 5)
        self.assertEqual(parsed['center']['y'], 10)
        self.assertEqual(parsed['radius'], 3.5)
        self.assertEqual(parsed['color'], 'BLUE')

    def test_union_type_encoding(self):
        """Test encoding of union types - this is where __dataclass_type__ is crucial."""
        # Create different shapes
        circle = Circle(Point(0, 0), 5.0, Color.RED)
        rectangle = Rectangle(Point(10, 10), 20, 15, Color.GREEN)
        triangle = Triangle(Point(0, 0), Point(10, 0), Point(5, 10), Color.BLUE)

        # Create shapes with union types
        shape1 = Shape(circle, Size.LARGE, "My Circle")
        shape2 = Shape(rectangle, Size.MEDIUM, "My Rectangle")
        shape3 = Shape(triangle, Size.SMALL, "My Triangle")

        # Encode each shape
        encoder = DataclassEncoder()
        for shape in [shape1, shape2, shape3]:
            result = json.dumps(shape, cls=DataclassEncoder)
            parsed = json.loads(result)

            self.assertEqual(parsed['__dataclass_type__'], 'Shape')
            self.assertIn('__dataclass_type__', parsed['shape'])
            self.assertEqual(parsed['size'], shape.size.name)
            self.assertEqual(parsed['name'], shape.name)

    def test_complex_nested_union_encoding(self):
        """Test encoding of complex nested structures with multiple unions."""
        circle = Circle(Point(1, 2), 3.0, Color.RED)
        rectangle = Rectangle(Point(4, 5), 6, 7, Color.GREEN)
        triangle = Triangle(Point(8, 9), Point(10, 11), Point(12, 13), Color.BLUE)
        point = Point(14, 15)

        complex_obj = ComplexNested(
            primary_shape=circle, secondary_shapes=[triangle, point], metadata="test metadata", config=Color.BLUE)

        result = json.dumps(complex_obj, cls=DataclassEncoder)
        parsed = json.loads(result)

        self.assertEqual(parsed['__dataclass_type__'], 'ComplexNested')
        self.assertEqual(parsed['primary_shape']['__dataclass_type__'], 'Circle')
        self.assertEqual(len(parsed['secondary_shapes']), 2)
        self.assertEqual(parsed['secondary_shapes'][0]['__dataclass_type__'], 'Triangle')
        self.assertEqual(parsed['secondary_shapes'][1]['__dataclass_type__'], 'Point')
        self.assertEqual(parsed['metadata'], "test metadata")
        self.assertEqual(parsed['config'], 'BLUE')

    def test_simple_decoding(self):
        """Test decoding of simple dataclass."""
        data = {'x': 15, 'y': 25, '__dataclass_type__': 'Point'}

        decoder = dataclass_decoder(Point)
        result = decoder(data)

        self.assertIsInstance(result, Point)
        self.assertEqual(result.x, 15)
        self.assertEqual(result.y, 25)

    def test_enum_decoding(self):
        """Test decoding of dataclass with enum fields."""
        data = {
            'center': {
                'x': 3,
                'y': 4,
                '__dataclass_type__': 'Point'
            },
            'radius': 2.5,
            'color': 'GREEN',
            '__dataclass_type__': 'Circle'
        }

        decoder = dataclass_decoder(Circle)
        result = decoder(data)

        self.assertIsInstance(result, Circle)
        self.assertIsInstance(result.center, Point)
        self.assertEqual(result.center.x, 3)
        self.assertEqual(result.center.y, 4)
        self.assertEqual(result.radius, 2.5)
        self.assertEqual(result.color, Color.GREEN)

    def test_round_trip_serialization(self):
        """Test complete round-trip serialization and deserialization."""
        # Create complex object
        circle = Circle(Point(1, 2), 3.0, Color.RED)
        rectangle = Rectangle(Point(4, 5), 6, 7, Color.GREEN)
        shapes = [Shape(circle, Size.LARGE, "Circle Shape"), Shape(rectangle, Size.MEDIUM, "Rectangle Shape")]
        drawing = Drawing(shapes, "Test Drawing", Color.BLUE)

        # Serialize to JSON string
        json_str = json.dumps(drawing, cls=DataclassEncoder)

        # Deserialize back
        json_data = json.loads(json_str)
        decoder = dataclass_decoder(Drawing)
        restored_drawing = decoder(json_data)

        # Verify structure
        self.assertIsInstance(restored_drawing, Drawing)
        self.assertEqual(restored_drawing.title, "Test Drawing")
        self.assertEqual(restored_drawing.background_color, Color.BLUE)
        self.assertEqual(len(restored_drawing.shapes), 2)

        # Verify first shape
        shape1 = restored_drawing.shapes[0]
        self.assertIsInstance(shape1, Shape)
        self.assertEqual(shape1.name, "Circle Shape")
        self.assertEqual(shape1.size, Size.LARGE)
        self.assertIsInstance(shape1.shape, Circle)

        # Verify second shape
        shape2 = restored_drawing.shapes[1]
        self.assertIsInstance(shape2, Shape)
        self.assertEqual(shape2.name, "Rectangle Shape")
        self.assertEqual(shape2.size, Size.MEDIUM)
        self.assertIsInstance(shape2.shape, Rectangle)

    def test_file_save_and_load(self):
        """Test saving to and loading from files."""
        # Create test object
        point = Point(100, 200)
        circle = Circle(point, 50.0, Color.RED)

        # Save to file
        filepath = os.path.join(self.temp_dir, "test_circle.json")
        save_to_json(circle, filepath)

        # Verify file exists and has content
        self.assertTrue(os.path.exists(filepath))
        with open(filepath, 'r') as f:
            content = f.read()
            self.assertIn('__dataclass_type__', content)
            self.assertIn('Circle', content)

        # Load from file
        loaded_circle = load_from_json(Circle, filepath)

        # Verify loaded object
        self.assertIsInstance(loaded_circle, Circle)
        self.assertEqual(loaded_circle.radius, 50.0)
        self.assertEqual(loaded_circle.color, Color.RED)
        self.assertIsInstance(loaded_circle.center, Point)
        self.assertEqual(loaded_circle.center.x, 100)
        self.assertEqual(loaded_circle.center.y, 200)

    def test_optional_fields(self):
        """Test serialization with optional fields."""
        # Test with None optional field
        obj1 = OptionalFields("required", None, Color.BLUE)
        json_str1 = json.dumps(obj1, cls=DataclassEncoder)
        parsed1 = json.loads(json_str1)

        decoder = dataclass_decoder(OptionalFields)
        restored1 = decoder(parsed1)
        self.assertEqual(restored1.required_field, "required")
        self.assertIsNone(restored1.optional_shape)
        self.assertEqual(restored1.default_color, Color.BLUE)

        # Test with actual optional field
        circle = Circle(Point(1, 1), 1.0, Color.GREEN)
        obj2 = OptionalFields("required2", circle)
        json_str2 = json.dumps(obj2, cls=DataclassEncoder)
        parsed2 = json.loads(json_str2)

        restored2 = decoder(parsed2)
        self.assertEqual(restored2.required_field, "required2")
        self.assertIsInstance(restored2.optional_shape, Circle)
        self.assertEqual(restored2.default_color, Color.RED)  # default value

    def test_list_of_unions(self):
        """Test serialization of lists containing union types."""
        circle = Circle(Point(0, 0), 1.0, Color.RED)
        rectangle = Rectangle(Point(1, 1), 2, 3, Color.GREEN)

        complex_obj = ComplexNested(
            primary_shape=circle,
            secondary_shapes=[
                Triangle(Point(0, 0), Point(1, 0), Point(0, 1), Color.BLUE),
                Point(10, 20),
                Triangle(Point(2, 2), Point(3, 2), Point(2, 3), Color.RED)
            ],
            metadata=42,  # int instead of str
            config=Size.LARGE)

        json_str = json.dumps(complex_obj, cls=DataclassEncoder)
        parsed = json.loads(json_str)

        # Verify the JSON structure contains type information
        self.assertEqual(parsed['__dataclass_type__'], 'ComplexNested')
        self.assertEqual(parsed['primary_shape']['__dataclass_type__'], 'Circle')
        self.assertEqual(len(parsed['secondary_shapes']), 3)
        self.assertEqual(parsed['secondary_shapes'][0]['__dataclass_type__'], 'Triangle')
        self.assertEqual(parsed['secondary_shapes'][1]['__dataclass_type__'], 'Point')
        self.assertEqual(parsed['secondary_shapes'][2]['__dataclass_type__'], 'Triangle')
        self.assertEqual(parsed['metadata'], 42)
        self.assertEqual(parsed['config'], 'LARGE')

    def test_json_encoder_error_handling(self):
        """Test that the encoder handles non-serializable objects appropriately."""
        encoder = DataclassEncoder()

        # Test with a non-dataclass, non-enum object
        class NonSerializable:
            pass

        obj = NonSerializable()

        # Should raise TypeError for non-serializable objects
        with self.assertRaises(TypeError):
            encoder.default(obj)

    def test_nested_enums_in_dataclass(self):
        """Test encoding/decoding of dataclasses with multiple enum fields."""

        @dataclass
        class MultiEnumClass:
            color: Color
            size: Size
            name: str

        obj = MultiEnumClass(Color.GREEN, Size.MEDIUM, "test")

        # Encode
        json_str = json.dumps(obj, cls=DataclassEncoder)
        parsed = json.loads(json_str)

        self.assertEqual(parsed['color'], 'GREEN')
        self.assertEqual(parsed['size'], 'MEDIUM')
        self.assertEqual(parsed['name'], 'test')

        # Decode
        decoder = dataclass_decoder(MultiEnumClass)
        restored = decoder(parsed)

        self.assertEqual(restored.color, Color.GREEN)
        self.assertEqual(restored.size, Size.MEDIUM)
        self.assertEqual(restored.name, "test")

    def test_empty_collections(self):
        """Test serialization with empty lists and None values."""
        empty_drawing = Drawing([], "Empty Drawing", Color.RED)

        json_str = json.dumps(empty_drawing, cls=DataclassEncoder)
        parsed = json.loads(json_str)

        decoder = dataclass_decoder(Drawing)
        restored = decoder(parsed)

        self.assertEqual(len(restored.shapes), 0)
        self.assertEqual(restored.title, "Empty Drawing")
        self.assertEqual(restored.background_color, Color.RED)

    def test_dict_with_union_values(self):
        """Test serialization of dictionaries with union type values."""
        # Create various shapes
        circle = Circle(Point(1, 2), 3.0, Color.RED)
        rectangle = Rectangle(Point(4, 5), 6, 7, Color.GREEN)
        triangle = Triangle(Point(8, 9), Point(10, 11), Point(12, 13), Color.BLUE)

        # Create dictionary container with union values
        dict_container = DictContainer(
            shape_registry={
                "my_circle": circle,
                "my_rectangle": rectangle,
                "my_triangle": triangle
            },
            config_map={
                Color.RED: "primary",
                Color.GREEN: 42,
                Color.BLUE: "secondary"
            },
            name="Test Container")

        # Serialize to JSON
        json_str = json.dumps(dict_container, cls=DataclassEncoder)
        parsed = json.loads(json_str)

        # Verify the JSON structure contains type information for union values
        self.assertEqual(parsed['__dataclass_type__'], 'DictContainer')
        self.assertEqual(parsed['name'], "Test Container")

        # Check shape registry
        shape_registry = parsed['shape_registry']
        self.assertEqual(shape_registry['my_circle']['__dataclass_type__'], 'Circle')
        self.assertEqual(shape_registry['my_rectangle']['__dataclass_type__'], 'Rectangle')
        self.assertEqual(shape_registry['my_triangle']['__dataclass_type__'], 'Triangle')

        # Check config map - enum keys should be serialized as names
        config_map = parsed['config_map']
        self.assertIn('RED', config_map)
        self.assertIn('GREEN', config_map)
        self.assertIn('BLUE', config_map)
        self.assertEqual(config_map['RED'], "primary")
        self.assertEqual(config_map['GREEN'], 42)
        self.assertEqual(config_map['BLUE'], "secondary")

        # Test round-trip deserialization
        decoder = dataclass_decoder(DictContainer)
        restored_container = decoder(parsed)

        # Verify the restored object
        self.assertIsInstance(restored_container, DictContainer)
        self.assertEqual(restored_container.name, "Test Container")

        # Verify shape registry
        self.assertEqual(len(restored_container.shape_registry), 3)
        self.assertIsInstance(restored_container.shape_registry['my_circle'], Circle)
        self.assertIsInstance(restored_container.shape_registry['my_rectangle'], Rectangle)
        self.assertIsInstance(restored_container.shape_registry['my_triangle'], Triangle)

        # Verify specific shape properties
        restored_circle = restored_container.shape_registry['my_circle']
        self.assertEqual(restored_circle.radius, 3.0)
        self.assertEqual(restored_circle.color, Color.RED)
        self.assertEqual(restored_circle.center.x, 1)
        self.assertEqual(restored_circle.center.y, 2)

        # Verify config map
        self.assertEqual(len(restored_container.config_map), 3)
        self.assertEqual(restored_container.config_map[Color.RED], "primary")
        self.assertEqual(restored_container.config_map[Color.GREEN], 42)
        self.assertEqual(restored_container.config_map[Color.BLUE], "secondary")


if __name__ == '__main__':
    unittest.main()
