'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.addColumn(
      'processes',
      'result',
      {
        type:Sequelize.UUID,
        onDelete: 'cascade',
        onUpdate: 'cascade',
        allowNull: true,  
        references: {
            model: 'files_mains',
            key: 'id',
            deferrable: Sequelize.Deferrable.INITIALLY_DEFERRED,
            onDelete: 'cascade',
            onUpdate: 'cascade',
        },
      }
    )
    /*
      Add altering commands here.
      Return a promise to correctly handle asynchronicity.

      Example:
      return queryInterface.createTable('users', { id: Sequelize.INTEGER });
    */
  },

  down: (queryInterface, Sequelize) => {
    /*
      Add reverting commands here.
      Return a promise to correctly handle asynchronicity.

      Example:
      return queryInterface.dropTable('users');
    */
  }
};
